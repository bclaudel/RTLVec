import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
import zipfile

from import_deepcircuitx import discover_repository, instantiated, read_index, extract_archive


class DeepCircuitXTests(unittest.TestCase):
    def test_archive_records_commit_and_skips_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'source.tar.gz'
            with tarfile.open(archive, 'w:gz', pax_headers={'comment': 'a' * 40}) as tar:
                content = b'module test(input a, output q); assign q=a; endmodule'
                member = tarfile.TarInfo('repo-HEAD/rtl/test.v'); member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
                link = tarfile.TarInfo('repo-HEAD/link'); link.type = tarfile.SYMTYPE
                link.linkname = '/etc/passwd'; tar.addfile(link)
            target = root / 'checkout'
            self.assertEqual(extract_archive(archive, target), 'a' * 40)
            self.assertEqual((target / 'rtl/test.v').read_bytes(), content)
            self.assertFalse((target / 'link').exists())

    def test_archive_rejects_path_traversal_and_missing_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, header, member_name in (
                ('escape', {'comment': 'a' * 40}, 'repo-HEAD/../outside'),
                ('unidentified', {}, 'repo-HEAD/test.v')):
                archive = root / (name + '.tar.gz')
                with tarfile.open(archive, 'w:gz', pax_headers=header) as tar:
                    member = tarfile.TarInfo(member_name); member.size = 1
                    tar.addfile(member, io.BytesIO(b'x'))
                with self.assertRaises(ValueError):
                    extract_archive(archive, root / 'checkout')
            self.assertFalse((root / 'outside').exists())

    def test_index_deduplicates_repositories_and_retains_categories(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("download_source_data/new_module_url.json", json.dumps([
                {"keyword": "Adder", "url_repo": ["https://github.com/Owner/RTL", "https://example.org/x"]},
                {"keyword": "Counter", "url_repo": ["https://github.com/owner/rtl.git"]}]))
            archive.writestr("download_source_data/encoder_decoder.json", "[]")
        entries = read_index(buffer.getvalue())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["categories"], ["Adder", "Counter"])
        self.assertEqual(entries, read_index(buffer.getvalue()))

    def test_nested_parameter_instances_are_dependencies(self):
        body = "leaf #(.WIDTH(2*(3+1))) u_leaf(.a(a)); leaf_signal = a;"
        self.assertEqual(instantiated(body, {"leaf": [], "leaf_signal": []}), {"leaf"})

    def test_all_collections_merge_urls_and_record_membership(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for filename in ('new_module_url.json', 'encoder_decoder.json', 'new_clip_url.json', 'new_ip_url.json'):
                archive.writestr('download_source_data/' + filename, json.dumps([
                    dict(keyword=filename, url_repo=['https://github.com/owner/shared'])]))
            archive.writestr('download_source_data/riscv_url.txt',
                             'https://github.com/owner/shared\nhttps://github.com/owner/cpu\n')
        entries = read_index(buffer.getvalue(), ('module', 'chip', 'ip', 'riscv'))
        self.assertEqual(len(entries), 2)
        shared = next(r for r in entries if r['url'].endswith('/shared'))
        self.assertEqual(shared['collections'], ['module', 'chip', 'ip', 'riscv'])
        self.assertIn('RISC-V', shared['categories'])

    def test_discovery_resolves_hierarchy_headers_and_excludes_testbenches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "rtl").mkdir()
            (root / "tb").mkdir()
            (root / "rtl/top.v").write_text('`include "width.vh"\n'
                '/* module fake(input a); endmodule */\n'
                'module top(input a, output q); leaf #(.W((2+2))) u(.a(a), .q(q)); endmodule')
            (root / "rtl/leaf.v").write_text('module leaf #(parameter W=4)(input a, output q); '
                                            'assign q=~a; endmodule')
            (root / "rtl/width.vh").write_text('`define WIDTH 4\n')
            (root / "tb/top.v").write_text('module top(input a, output q); endmodule')
            record = dict(checkout=str(root), url="https://github.com/owner/repo", commit="123",
                          categories=["Counter"], license_files=["LICENSE"])
            designs, rejected = discover_repository(record)
            self.assertFalse(rejected)
            self.assertEqual({d["top"] for d in designs}, {"top", "leaf"})
            top = next(d for d in designs if d["top"] == "top")
            self.assertEqual(set(top["sources"]), {str(root / "rtl/top.v"), str(root / "rtl/leaf.v")})
            self.assertEqual(top["include_dirs"], [str(root / "rtl")])
            self.assertEqual(top["project_family"], "owner/repo")
            self.assertEqual(json.loads(top["source_provenance"])["commit"], "123")
            limited, _ = discover_repository(record, max_modules=1)
            unlimited, _ = discover_repository(record, max_modules=0)
            self.assertEqual(len(limited), 1)
            self.assertEqual(len(unlimited), 2)

    def test_ambiguous_definitions_and_escaping_includes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for file in ("a.v", "b.v"):
                (root / file).write_text('module same(input a, output q); assign q=a; endmodule')
            (root / "top.v").write_text('`include "../outside.vh"\n'
                                        'module top(input a, output q); assign q=~a; endmodule')
            record = dict(checkout=str(root), url="https://github.com/owner/repo", commit="123",
                          categories=[], license_files=[])
            designs, rejected = discover_repository(record)
            self.assertFalse(designs)
            self.assertEqual(len(rejected), 2)


if __name__ == "__main__":
    unittest.main()
