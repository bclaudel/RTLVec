"""Convert synthesizable Verilog to our versioned RTL graphs and assemble datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from dataset import file_hash, json_hash, load_datasets, validate_graph, dataset_statistics, validate_contract
from rtl_graph import GRAPH_SPEC, PASSES, graph_from_json, preprocessing_contract, typed_graph_hash, graphs_equal


def validate_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("designs"), list) or not manifest["designs"]:
        raise ValueError("manifest requires schema_version=1 and a nonempty designs list")
    records, seen = [], set()
    for index, design in enumerate(manifest["designs"]):
        record = {"index": index, "design_id": design.get("design_id") if isinstance(design, dict) else None}
        try:
            if not isinstance(design, dict):
                raise ValueError("design must be an object")
            for key in ("design_id", "project_family", "top", "source_provenance"):
                if not isinstance(design.get(key), str) or not design[key].strip():
                    raise ValueError(f"{key} must be a nonempty string")
            if design["design_id"] in seen:
                raise ValueError("duplicate design_id")
            seen.add(design["design_id"])
            sources = design.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ValueError("sources must be a nonempty list")
            hashes = {}
            for source in sources:
                if not isinstance(source, str):
                    raise ValueError("source paths must be strings")
                full = (path.parent / source).resolve()
                if full.suffix.lower() not in (".v", ".vh") or not full.is_file():
                    raise ValueError(f"missing or unsupported Verilog source: {source}")
                hashes[source] = file_hash(full)
            include_hashes = {}
            header_hashes = {}
            for source in sources:
                for header in (path.parent / source).resolve().parent.glob("*.vh"):
                    header_hashes[os.path.relpath(header, path.parent)] = file_hash(header)
            includes = design.get("include_dirs", [])
            if not isinstance(includes, list):
                raise ValueError("include_dirs must be a list")
            for name in includes:
                full = (path.parent / name).resolve()
                if not full.is_dir():
                    raise ValueError(f"missing include directory: {name}")
                # Include contents affect preprocessing, not just listed source files.
                include_hashes[name] = {str(p.relative_to(full)): file_hash(p)
                                       for p in sorted(full.rglob("*")) if p.is_file()}
            for key in ("defines", "parameters"):
                values = design.get(key, {})
                if not isinstance(values, dict) or not all(
                    isinstance(k, str) and k and isinstance(v, (str, int, float, bool))
                    and (not isinstance(v, float) or math.isfinite(v)) for k, v in values.items()
                ):
                    raise ValueError(f"{key} must map names to finite scalar values")
            config = {"sources": sources, "top": design["top"], "include_dirs": includes,
                      "defines": design.get("defines", {}), "parameters": design.get("parameters", {})}
            record.update(status="accepted", project_family=design["project_family"],
                          source_provenance=design["source_provenance"], source_hashes=hashes,
                          include_hashes=include_hashes, configuration=config,
                          header_hashes=header_hashes,
                          source_configuration_hash=json_hash({"config": config, "sources": hashes,
                                                               "includes": include_hashes, "headers": header_hashes}))
        except (ValueError, TypeError, OSError) as exc:
            record.update(status="rejected", error=str(exc))
        records.append(record)
    return {"schema_version": 1, "manifest": str(path), "designs": records,
            "accepted": sum(r["status"] == "accepted" for r in records),
            "rejected": sum(r["status"] == "rejected" for r in records),
            "graph_schema": GRAPH_SPEC["name"]}



def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def yosys_executable(explicit=None):
    executable = explicit or shutil.which("yosys") or shutil.which("yowasp-yosys")
    if not executable:
        candidate = Path(sys.executable).parent / "yowasp-yosys"
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise ValueError("Yosys is missing; run uv sync or supply --yosys /path/to/yosys")
    return str(Path(shutil.which(executable) or executable).resolve())


def yosys_version(executable, timeout):
    result = subprocess.run([executable, "-V"], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError(f"Yosys version check failed: {result.stderr.strip()}")
    return result.stdout.strip()


def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", value):
        raise ValueError(f"Unsupported Verilog identifier: {value!r}")
    return value


def scalar(value):
    text = str(int(value)) if isinstance(value, bool) else str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.$'()+*/<>=!&|^~?:-]+", text):
        raise ValueError(f"Only simple numeric/macro expressions are supported: {value!r}")
    return text


class YosysError(ValueError):
    def __init__(self, log):
        self.log = log
        super().__init__(log[-6000:])


def run_yosys(executable, record, manifest_dir, workspace, timeout):
    config = record["configuration"]
    sources = [(manifest_dir / name).resolve() for name in config["sources"]]
    includes = [(manifest_dir / name).resolve() for name in config["include_dirs"]]
    root = Path(os.path.commonpath([str(manifest_dir), *map(str, (p.parent for p in sources)), *map(str, includes)]))
    # Compile a snapshot of declared inputs so undeclared include dependencies
    # fail rather than silently escaping cache identity.
    inputs = workspace / "inputs"
    inputs.mkdir()
    files = {**record["source_hashes"], **record["header_hashes"]}
    for include_dir, hashes in record["include_hashes"].items():
        for name, digest in hashes.items():
            files[str(Path(include_dir) / name)] = digest
    for name, digest in files.items():
        source = (manifest_dir / name).resolve()
        if file_hash(source) != digest:
            raise ValueError(f"Input changed during conversion: {name}")
        destination = inputs / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_hash(destination) != digest:
            raise ValueError(f"Input changed while snapshotting: {name}")
        # Absolute includes cannot be relocated reproducibly into the snapshot.
        if destination.suffix in (".v", ".vh") and re.search(r'`include\s+"/', destination.read_text(errors="replace")):
            raise ValueError("Absolute RTL includes are unsupported; use declared include_dirs")
    for include in includes:
        (inputs / include.relative_to(root)).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    is_wasm = "yowasp" in Path(executable).name
    if is_wasm:
        if any(character in str(root) + str(workspace) for character in (":", "=")):
            raise ValueError("YoWASP input paths cannot contain ':' or '='")
        env["YOWASP_MOUNT"] = f"/src={inputs}:/work={workspace}"
    def source_path(path):
        return str((Path("/src") if is_wasm else inputs) / path.relative_to(root))
    def work_path(name):
        return f"/work/{name}" if is_wasm else str(workspace / name)
    quote = json.dumps
    # Source directories are included explicitly for adjacent header files.
    flags = []
    for index, directory in enumerate(sorted(set(includes + [p.parent for p in sources]))):
        alias = f"include_{index}"
        shutil.copytree(inputs / directory.relative_to(root), workspace / alias)
        flag_path = work_path(alias)
        if any(character.isspace() for character in flag_path):
            raise ValueError("The temporary directory path must not contain whitespace")
        flags.append("-I" + flag_path)
    flags += ["-D" + identifier(name) + "=" + scalar(value) for name, value in sorted(config["defines"].items())]
    top = identifier(config["top"])
    commands = ["read_verilog " + " ".join(flags + [quote(source_path(path)) for path in sources])]
    if config["parameters"]:
        settings = " ".join("-set " + identifier(name) + " " + scalar(value)
                            for name, value in sorted(config["parameters"].items()))
        commands.append(f"chparam {settings} {top}")
    commands += [f"hierarchy -check -top {top}", *PASSES, "write_json " + quote(work_path("netlist.json"))]
    script = workspace / "run.ys"
    script.write_text("\n".join(commands) + "\n")
    result = subprocess.run([executable, "-s", work_path("run.ys")], cwd=workspace, env=env,
                            capture_output=True, text=True, timeout=timeout)
    log = result.stdout + result.stderr
    if result.returncode:
        raise YosysError(log)
    return json.loads((workspace / "netlist.json").read_text()), log


def convert_manifest(path, output, *, yosys=None, timeout=120, max_nodes=2048, jobs=1):
    if timeout <= 0 or not math.isfinite(timeout) or max_nodes < 2 or jobs < 1:
        raise ValueError("timeout must be finite and positive, max_nodes must be >=2, jobs must be >=1")
    path, output = Path(path).resolve(), Path(output).resolve()
    report = validate_manifest(path)
    executable = yosys_executable(yosys)
    contract = preprocessing_contract(yosys_version(executable, timeout))
    pre_hash = json_hash(contract)
    output.mkdir(parents=True, exist_ok=True)
    (output / "graphs").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    def convert_record(record):
        if record["status"] != "accepted":
            return record
        entry = dict(record)
        key = json_hash({"design_id": record["design_id"],
                         "project_family": record["project_family"],
                         "source_configuration_hash": record["source_configuration_hash"],
                         "preprocessing_fingerprint": pre_hash})
        artifact = output / "graphs" / f"{key}.pt"
        metadata_path = artifact.with_suffix(".json")
        log_path = output / "logs" / f"{key}.log"
        try:
            cached = False
            if artifact.exists() and metadata_path.exists():
                cached_metadata = json.loads(metadata_path.read_text())
                cached = (cached_metadata.get("file_hash") == file_hash(artifact)
                          and cached_metadata.get("preprocessing_fingerprint") == pre_hash)
            if cached:
                graph = torch.load(artifact, weights_only=False, map_location="cpu")
                validate_graph(graph)
                if graph.num_nodes > max_nodes:
                    raise ValueError(f"Cached graph exceeds max_nodes={max_nodes}")
            else:
                with tempfile.TemporaryDirectory(prefix="structrtl-") as temp:
                    netlist, log = run_yosys(executable, record, path.parent, Path(temp), timeout)
                    log_path.write_text(log)
                    graph = graph_from_json(netlist, record["configuration"]["top"], max_nodes=max_nodes)
                validate_graph(graph)
                graph.design_id = record["design_id"]
                graph.project_family = record["project_family"]
                temporary = artifact.with_suffix(".pt.tmp")
                torch.save(graph, temporary)
                temporary.replace(artifact)
                write_json(artifact.with_suffix(".netlist.json"), netlist)
                write_json(metadata_path, {"file_hash": file_hash(artifact),
                                           "preprocessing_fingerprint": pre_hash})
            entry.update(graph_file=str(artifact.relative_to(output)), file_hash=file_hash(artifact),
                         typed_graph_hash=typed_graph_hash(graph), cached=cached,
                         log_file=str(log_path.relative_to(output)))
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(str(exc.stdout or "") + str(exc.stderr or ""))
            entry.update(status="rejected", error=f"Yosys exceeded timeout={timeout}s", log_file=str(log_path.relative_to(output)))
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            log_path.write_text(getattr(exc, "log", str(exc)))
            entry.update(status="rejected", error=str(exc), log_file=str(log_path.relative_to(output)))
        return entry
    records = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for index, entry in enumerate(executor.map(convert_record, report["designs"]), 1):
            records.append(entry)
            if index % 25 == 0 or index == len(report["designs"]):
                print(f"Converted {index}/{len(report['designs'])}: "
                      f"{sum(r['status'] == 'accepted' for r in records)} accepted", flush=True)
    report.update(schema_version=2, preprocessing=contract, preprocessing_fingerprint=pre_hash,
                  designs=records, accepted=sum(r["status"] == "accepted" for r in records),
                  rejected=sum(r["status"] == "rejected" for r in records))
    write_json(output / "conversion.json", report)
    return report


def assign_families(families, previous=None):
    assignments = dict(previous or {})
    if any(split not in ("train", "val") for split in assignments.values()):
        raise ValueError("Invalid previous family assignment")
    def score(family):
        return int(json_hash({"seed": 42, "family": family}), 16)
    ordered = sorted(set(families), key=lambda family: (score(family), family))
    if not assignments:
        if len(ordered) < 2:
            raise ValueError("At least two distinct, nonduplicate project families are needed for train/validation")
        validation_count = max(1, round(len(ordered) * 0.2))
        assignments = {family: "val" if i < validation_count else "train" for i, family in enumerate(ordered)}
    else:
        for family in ordered:
            assignments.setdefault(family, "val" if score(family) / 2**256 < 0.2 else "train")
    return assignments


def assemble_dataset(input_dir, output, *, allow_rejected=False, split_from=None):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    report = json.loads((input_dir / "conversion.json").read_text())
    validate_contract(report)
    if report.get("rejected") and not allow_rejected:
        raise ValueError("Conversion report has rejected designs; fix the manifest and rerun conversion before assembly")
    if not report.get("designs"):
        raise ValueError("No converted designs to assemble")
    previous = {}
    existing_ids = set()
    if (output / "dataset.json").exists():
        existing = json.loads((output / "dataset.json").read_text())
        validate_contract(existing)
        if existing["preprocessing_fingerprint"] != report["preprocessing_fingerprint"]:
            raise ValueError("Existing output uses different preprocessing; choose a new output directory")
        previous = existing["family_splits"]
        existing_ids = {entry["design_id"] for entry in existing["designs"]}
    elif any((output / f"{split}.pt").exists() for split in ("train", "val")):
        raise ValueError("Refusing to overwrite existing graph files without our dataset manifest")
    elif split_from is not None:
        # Reuse identities/assignments, never graph tensors from another toolchain.
        _, _, reference = load_datasets(split_from)
        previous = reference["family_splits"]
        existing_ids = {entry["design_id"] for entry in reference["designs"]}
    buckets, accepted, duplicates, ids, source_keys = {}, [], [], set(), {}
    for record in sorted(report["designs"], key=lambda r: (r["design_id"] not in existing_ids, r["design_id"])):
        if record["status"] == "rejected" and allow_rejected:
            continue
        if record["status"] != "accepted" or record["design_id"] in ids:
            raise ValueError("Conversion report contains an invalid/duplicate design")
        ids.add(record["design_id"])
        if len(ids) % 500 == 0:
            print(f"Assembly: {len(ids)} accepted candidates checked, {len(accepted)} unique graphs, {len(duplicates)} duplicates", flush=True)
        artifact = (input_dir / record["graph_file"]).resolve()
        if not artifact.is_relative_to(input_dir) or file_hash(artifact) != record["file_hash"]:
            raise ValueError(f"Graph file mismatch: {record['design_id']}")
        graph = torch.load(artifact, weights_only=False, map_location="cpu")
        validate_graph(graph)
        if graph.design_id != record["design_id"] or graph.project_family != record["project_family"]:
            raise ValueError("Graph identity does not match conversion report")
        graph_hash = typed_graph_hash(graph)
        if graph_hash != record["typed_graph_hash"]:
            raise ValueError("Typed graph fingerprint mismatch")
        duplicate_of = source_keys.get(record["source_configuration_hash"])
        if not duplicate_of:
            for previous_record, previous_graph in buckets.get(graph_hash, []):
                if graphs_equal(graph, previous_graph):
                    duplicate_of = previous_record["design_id"]
                    break
        if duplicate_of:
            duplicates.append({"design_id": record["design_id"], "duplicate_of": duplicate_of})
            continue
        buckets.setdefault(graph_hash, []).append((record, graph))
        source_keys[record["source_configuration_hash"]] = record["design_id"]
        accepted.append((record, graph))
    assignments = assign_families([r["project_family"] for r, _ in accepted], previous)
    datasets, records = {"train": [], "val": []}, []
    for record, graph in accepted:
        split = assignments[record["project_family"]]
        entry = {**{key: value for key, value in record.items() if key not in ("cached", "log_file", "index")},
                 "split": split, "index": len(datasets[split])}
        datasets[split].append(graph)
        records.append(entry)
    if any(not graphs for graphs in datasets.values()):
        raise ValueError("Assembly would produce an empty split; add a distinct project family")
    metadata = dict(schema_version=2, preprocessing=report["preprocessing"],
                    preprocessing_fingerprint=report["preprocessing_fingerprint"],
                    family_splits=assignments, designs=records, duplicates=duplicates,
                    statistics=dataset_statistics(datasets))
    metadata["conversion_summary"] = {
        "candidates": len(report["designs"]), "accepted": len(accepted),
        "duplicates": len(duplicates), "rejected": report.get("rejected", 0)}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assembly-", dir=output) as temp:
        stage = Path(temp)
        for split, graphs in datasets.items():
            torch.save(graphs, stage / f"{split}.pt")
        metadata["files"] = {split: file_hash(stage / f"{split}.pt") for split in datasets}
        metadata["dataset_fingerprint"] = json_hash({"files": metadata["files"],
            "preprocessing_fingerprint": metadata["preprocessing_fingerprint"],
            "designs": records, "family_splits": assignments})
        write_json(stage / "dataset.json", metadata)
        load_datasets(stage)
        # Commit the manifest last. Interrupted publication is detected by hashes.
        for name in ("train.pt", "val.pt", "dataset.json"):
            (stage / name).replace(output / name)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate RTL project metadata, not HDL semantics")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--report")
    convert = commands.add_parser("convert", help="lower RTL with Yosys and construct our graphs")
    convert.add_argument("--manifest", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--timeout", type=float, default=120)
    convert.add_argument("--max-nodes", type=int, default=2048)
    convert.add_argument("--yosys", help="native or YoWASP Yosys executable")
    convert.add_argument("--jobs", type=int, default=1, help="concurrent conversions")
    assemble = commands.add_parser("assemble", help="deduplicate and export grouped train/validation splits")
    assemble.add_argument("--input", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--allow-rejected", action="store_true",
                          help="assemble accepted designs; preserve failures in conversion.json")
    assemble.add_argument("--split-from", help="initial grouped split assignments from a validated reference dataset")
    inspect = commands.add_parser("inspect", help="validate our assembled dataset and its provenance")
    inspect.add_argument("--data-dir", default="data/expanded")
    inspect.add_argument("--report")
    args = parser.parse_args()
    try:
        if args.command == "convert":
            if args.jobs > 1:
                torch.set_num_threads(1)
            report = convert_manifest(args.manifest, args.output, yosys=args.yosys,
                                      timeout=args.timeout, max_nodes=args.max_nodes, jobs=args.jobs)
        elif args.command == "assemble":
            report = assemble_dataset(args.input, args.output, allow_rejected=args.allow_rejected,
                                      split_from=args.split_from)
        elif args.command == "validate":
            report = validate_manifest(args.manifest)
        else:
            _, _, report = load_datasets(args.data_dir)
        if getattr(args, "report", None):
            write_json(args.report, report)
        summary = {key: value for key, value in report.items() if key not in ("designs", "preprocessing")}
        if args.command in ("convert", "validate"):
            summary["designs"] = [{key: value for key, value in r.items()
                                   if key in ("design_id", "status", "error", "cached", "log_file")}
                                  for r in report["designs"]]
        print(json.dumps(summary, indent=2))
        return int(report.get("rejected", 0) > 0)
    except (ValueError, OSError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
