"""Fetch DeepCircuitX's published source repositories and discover RTL candidates.

Discovery is deliberately conservative; Yosys conversion decides HDL validity.
No upstream build scripts or testbenches are executed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

INDEX_URL = "https://drive.google.com/uc?export=download&id=1KUdJTL0LmEXtVJIA5vKKSxbNygxBQKsV"
SOURCE_PAGE = "https://zeju.gitbook.io/lcm-team/deepcircuitx/souce-rtl-code"
INDEX_FILES = ("new_module_url.json", "encoder_decoder.json")
COLLECTION_FILES = {
    "module": INDEX_FILES,
    "chip": ("new_clip_url.json",),
    "ip": ("new_ip_url.json",),
    "riscv": ("riscv_url.txt",),
}
NAME = r"[A-Za-z_][A-Za-z0-9_$]*"
MODULE = re.compile(r"\bmodule\s+(?:automatic\s+)?(" + NAME + r")\b(.*?)\bendmodule\b", re.S)
SKIP_DIRS = {".git", "tb", "test", "tests", "testbench", "testbenches", "sim", "simulation"}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def strip_comments(text):
    # Preserve strings (including include filenames), but hide comments.
    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else " " * len(m[0]), text, flags=re.S)


def read_index(content, collections=("module",)):
    repos = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for collection in collections:
          for filename in COLLECTION_FILES[collection]:
            raw = archive.read("download_source_data/" + filename)
            rows = ([{"keyword": "RISC-V", "url_repo": raw.decode().splitlines()}]
                    if filename.endswith(".txt") else json.loads(raw))
            for row in rows:
                for url in row["url_repo"]:
                    match = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?", url)
                    if not match:
                        continue
                    canonical = "https://github.com/" + "/".join(match.groups())
                    key = canonical.lower()
                    entry = repos.setdefault(key, {"url": canonical, "categories": [], "collections": []})
                    if collection not in entry["collections"]:
                        entry["collections"].append(collection)
                    if row["keyword"] not in entry["categories"]:
                        entry["categories"].append(row["keyword"])
    # Deterministic round robin across categories avoids selecting only adders.
    categories = {}
    for key, entry in repos.items():
        categories.setdefault(sorted(entry["categories"])[0], []).append(key)
    queues = [sorted(categories[c], key=digest) for c in sorted(categories)]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(repos[queue.pop(0)])
    return ordered


def git(*args, cwd=None, timeout=90):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"})
    if result.returncode:
        raise ValueError(result.stderr.strip()[-1500:])
    return result.stdout.strip()


def extract_archive(archive, destination):
    """Extract regular files only, preserving the Git commit from the tar header."""
    with tarfile.open(archive, "r:gz") as source:
        commit = source.pax_headers.get("comment", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Source archive does not identify its exact Git commit")
        total = 0
        for member in source:
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise ValueError("Unsafe path in source archive")
            if not member.isfile() or len(parts) < 2:
                continue
            total += member.size
            if total > 2_000_000_000:
                raise ValueError("Source archive exceeds 2 GB expanded limit")
            path = destination.joinpath(*parts[1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, path.open("wb") as output:
                shutil.copyfileobj(stream, output)
        return commit


def fetch_archive(entry, destination, workspace):
    url = entry["url"].replace("https://github.com/", "https://codeload.github.com/") + "/tar.gz/HEAD"
    archive = workspace / "source.tar.gz"
    sha = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > 512_000_000:
                raise ValueError("Source archive exceeds 512 MB download limit")
            sha.update(chunk)
            output.write(chunk)
    destination.mkdir()
    metadata = {"commit": extract_archive(archive, destination), "archive_url": url,
                "source_archive_sha256": sha.hexdigest(), "transport": "archive"}
    (destination / ".deepcircuitx-source.json").write_text(json.dumps(metadata) + "\n")
    return metadata


def fetch_repository(entry, root, *, download=True, transport="git"):
    directory = root / "repositories" / digest(entry["url"].lower())[:16]
    try:
        marker = directory / ".deepcircuitx-source.json"
        if not (directory / ".git").is_dir() and not marker.is_file():
            if not download:
                raise ValueError("Repository not downloaded; see download-results.json for fetch failures")
            if directory.exists():
                raise ValueError("Incomplete checkout exists; move it aside before retrying")
            # Failed downloads leave no partial checkout at the persistent path.
            with tempfile.TemporaryDirectory(prefix="download-", dir=root) as temp:
                checkout = Path(temp) / "checkout"
                if transport == "archive":
                    fetch_archive(entry, checkout, Path(temp))
                else:
                    git("clone", "--depth", "1", "--quiet", entry["url"], str(checkout))
                checkout.rename(directory)
        metadata = json.loads(marker.read_text()) if marker.is_file() else {"commit": git("rev-parse", "HEAD", cwd=directory), "transport": "git"}
        commit = metadata["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Invalid pinned commit")
        licenses = [p for p in directory.iterdir()
                    if p.is_file() and not p.is_symlink()
                    and re.match(r"^(license|licence|copying)(\.|$)", p.name, re.I)]
        return {**entry, **metadata, "status": "downloaded", "checkout": str(directory), "commit": commit,
                "license_files": [str(p.relative_to(directory)) for p in sorted(licenses)]}
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        return {**entry, "status": "rejected", "error": str(exc)}


def instantiated(body, known):
    """Find named instances of known modules, including balanced #(...) parameters."""
    found = set()
    for match in re.finditer(r"\b(" + NAME + r")\b", body):
        name = match[0]
        if name not in known:
            continue
        tail = body[match.end():].lstrip()
        if tail.startswith("#"):
            tail = tail[1:].lstrip()
            if not tail.startswith("("):
                continue
            depth = 0
            for index, char in enumerate(tail):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    tail = tail[index + 1:].lstrip()
                    break
        if re.match(NAME + r"\s*(?:\[[^\]]*\]\s*)?\(", tail):
            found.add(name)
    return found


def discover_repository(record, max_modules=40):
    root = Path(record["checkout"])
    texts, declarations, headers = {}, {}, {}
    rejected = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in (".v", ".vh"):
            continue
        if any(part.lower() in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if path.stat().st_size > 2_000_000:
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        headers.setdefault(path.name, []).append(path)
        if path.suffix.lower() != ".v" or re.search(r"(?:^tb_|_tb\.|testbench)", path.name, re.I):
            continue
        text = strip_comments(path.read_text(errors="replace"))
        texts[path] = text
        for match in MODULE.finditer(text):
            declarations.setdefault(match[1], []).append((path, match[2]))
    # Resolve each file's dependencies once. Large libraries often declare
    # hundreds of modules in one file; rescanning it for every top is quadratic.
    dependencies = {path: instantiated(text, declarations) for path, text in texts.items()}
    closure_cache = {}
    def source_closure(source):
        if source in closure_cache:
            value = closure_cache[source]
            if isinstance(value, str):
                raise ValueError(value)
            return value
        sources, pending = set(), [source]
        while pending:
            path = pending.pop()
            if path in sources:
                continue
            sources.add(path)
            for dependency in dependencies[path]:
                choices = declarations[dependency]
                if len(choices) != 1:
                    error = f"Ambiguous dependency {dependency}"
                    closure_cache[source] = error
                    raise ValueError(error)
                pending.append(choices[0][0])
        closure_cache[source] = sources
        return sources
    designs = []
    for name in sorted(declarations, key=lambda n: digest(record["url"] + n)):
        if max_modules and len(designs) >= max_modules:
            break
        if len(declarations[name]) != 1:
            rejected.append({"top": name, "error": "Ambiguous module definition"})
            continue
        source, body = declarations[name][0]
        # Skip testbenches and helper blocks without a port declaration.
        if not re.search(r"\b(input|output|inout)\b", body):
            continue
        try:
            # Yosys reads the whole file, including sibling module declarations.
            sources = source_closure(source)
            includes, checked = set(), set()
            def include_dependencies(path):
                if path in checked:
                    return
                checked.add(path)
                text = strip_comments(path.read_text(errors="replace"))
                for include in re.findall(r'`include\s+"([^"\n]+)"', text):
                    if Path(include).is_absolute() or ".." in Path(include).parts:
                        raise ValueError(f"Unsafe include path: {include}")
                    candidates = [path.parent / include, root / include]
                    resolved = next((p for p in candidates if p.is_file() and not p.is_symlink()), None)
                    if resolved is None:
                        matches = headers.get(Path(include).name, [])
                        if len(matches) == 1 and include == Path(include).name:
                            resolved = matches[0]
                    if resolved is None or not resolved.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f"Missing/ambiguous include: {include}")
                    # The converter snapshots include directories, including .v includes.
                    includes.add(resolved.parent)
                    include_dependencies(resolved)
            for path in sorted(sources):
                include_dependencies(path)
            repo_name = record["url"].removeprefix("https://github.com/").lower()
            design_id = "deepcircuitx:" + repo_name + ":" + str(source.relative_to(root)) + ":" + name
            provenance = json.dumps({"collection": "DeepCircuitX", "source_page": SOURCE_PAGE,
                                     "repository": record["url"], "commit": record["commit"],
                                     "categories": record["categories"],
                                     "transport": record.get("transport", "git"),
                                     "source_archive_sha256": record.get("source_archive_sha256"),
                                     "license_files": record["license_files"]}, sort_keys=True)
            designs.append({"design_id": design_id, "project_family": repo_name,
                            "source_provenance": provenance, "top": name,
                            "sources": [str(p.resolve()) for p in sorted(sources)],
                            "include_dirs": [str(p.resolve()) for p in sorted(includes)]})
        except (ValueError, OSError, RecursionError) as exc:
            rejected.append({"top": name, "error": str(exc)})
    return designs, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/deepcircuitx")
    parser.add_argument("--index", help="local published URL-list ZIP; otherwise download once")
    parser.add_argument("--repos", type=int, default=100, help="deterministic repository prefix; 0 imports all")
    parser.add_argument("--modules-per-repo", type=int, default=40, help="candidate cap per repository; 0 disables cap")
    parser.add_argument("--collections", nargs="+", choices=[*COLLECTION_FILES, "all"], default=["module"])
    parser.add_argument("--jobs", type=int, default=8, help="concurrent repository downloads")
    parser.add_argument("--local-only", action="store_true", help="discover existing checkouts without network access")
    parser.add_argument("--transport", choices=["git", "archive"], default="git",
                        help="fetch missing checkouts with Git or GitHub source archives")
    args = parser.parse_args()
    if min(args.repos, args.modules_per_repo) < 0 or args.jobs < 1:
        parser.error("limits must be nonnegative and jobs must be positive")
    root = Path(args.output).resolve()
    (root / "repositories").mkdir(parents=True, exist_ok=True)
    index_path = root / "source-index.zip"
    if args.index:
        content = Path(args.index).read_bytes()
    elif index_path.exists():
        content = index_path.read_bytes()
    else:
        with urllib.request.urlopen(INDEX_URL, timeout=60) as response:
            content = response.read(5_000_001)
        if len(content) > 5_000_000:
            raise ValueError("Unexpectedly large source index")
    collections = list(COLLECTION_FILES) if "all" in args.collections else list(dict.fromkeys(args.collections))
    entries = read_index(content, collections)
    selected = entries[:args.repos] if args.repos else entries
    index_path.write_bytes(content)
    designs, records = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for i, record in enumerate(executor.map(lambda e: fetch_repository(e, root, download=not args.local_only, transport=args.transport), selected), 1):
            if record["status"] == "downloaded":
                try:
                    candidates, rejected = discover_repository(record, args.modules_per_repo)
                    record.update(candidates=len(candidates), discovery_rejections=rejected)
                    designs.extend(candidates)
                except (ValueError, OSError, RecursionError) as exc:
                    record.update(discovery_error=str(exc), candidates=0)
            records.append(record)
            print(f"[{i}/{len(selected)}] {record['url']}: {record['status']} "
                  f"({record.get('candidates', 0)} candidates)", flush=True)
            if i % 50 == 0:
                (root / "import-progress.json").write_text(json.dumps({
                    "processed_repositories": i, "selected_repositories": len(selected),
                    "candidate_designs": len(designs), "repositories": records}, indent=2) + "\n")
    report = {"schema_version": 1, "source_page": SOURCE_PAGE, "index_url": INDEX_URL,
              "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "index_sha256": hashlib.sha256(content).hexdigest(), "available_repositories": len(entries),
              "requested_repositories": len(selected), "modules_per_repo": args.modules_per_repo,
              "collections": collections,
              "repositories": records, "candidate_designs": len(designs)}
    (root / "import.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "projects.json").write_text(json.dumps({"schema_version": 1, "designs": designs}, indent=2) + "\n")
    print(f"Manifest: {root / 'projects.json'} ({len(designs)} candidates)")
    return int(not designs)


if __name__ == "__main__":
    raise SystemExit(main())
