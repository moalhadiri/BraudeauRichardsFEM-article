"""Publish a completed run with the repository owner's GitHub permission."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib


OWNER = "moalhadiri"
REPOSITORY = "BraudeauRichardsFEM-article"
BRANCH = "main"
SECRET_NAME = "BRAUDEAU_GITHUB_TOKEN"
API_ROOT = "https://api.github.com"
REPO_PATH = f"/repos/{OWNER}/{REPOSITORY}"
BEGIN = "<!-- BEGINEXAMPLERESULTS -->"
END = "<!-- ENDEXAMPLERESULTS -->"
FIGURES = ("spatial_errors_2d.png", "spatial_errors_3d.png", "temporal_errors.png", "solution_2d.png",
           "slotted_sphere.png", "dry_infiltration.png", "newton_corrections.png")


class PublicationError(RuntimeError):
    """A publication problem that leaves the numerical results untouched."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json(data):
    def invalid_constant(_value):
        raise ValueError("Non-finite JSON value")
    return json.loads(data, object_pairs_hook=_object, parse_constant=invalid_constant)


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is not None


def _digest_matches(data, expected):
    return isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None and _sha256(data) == expected


def _bounds(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) >= text.index(END):
        raise PublicationError("The README needs one complete computed-results block.")
    return text.index(BEGIN), text.index(END) + len(END)


def _check_png(data):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a PNG")
    position, first, image_data, ended = 8, True, False, False
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError("Incomplete PNG chunk")
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        end = position + 12 + length
        if end > len(data):
            raise ValueError("Incomplete PNG data")
        payload = data[position + 8:end - 4]
        checksum = struct.unpack(">I", data[end - 4:end])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != checksum:
            raise ValueError("Invalid PNG checksum")
        if first:
            if kind != b"IHDR" or length != 13:
                raise ValueError("Missing PNG header")
            width, height = struct.unpack(">II", payload[:8])
            if not width or not height:
                raise ValueError("Empty PNG")
            first = False
        if kind == b"IDAT" and length:
            image_data = True
        if kind == b"IEND":
            if length or end != len(data):
                raise ValueError("Invalid PNG ending")
            ended = True
        position = end
    if not image_data or not ended:
        raise ValueError("Incomplete PNG image")


def _prepare(project_root, output_dir):
    try:
        project = Path(project_root).resolve(strict=True)
        output = Path(output_dir).resolve(strict=True)
        metadata_bytes = (output / "latest_results.json").read_bytes()
        metadata = _json(metadata_bytes)
        cases, checks = metadata["cases"], metadata["checks"]
        count, checked = metadata["calculations"], metadata["checks_total"]
        if (metadata["status"] != "passed" or type(count) is not int or count <= 0 or type(checked) is not int
                or checked <= 0 or type(metadata["checks_passed"]) is not int or metadata["checks_passed"] != checked or len(cases) != count or len(checks) != checked
                or any(check["status"] != "passed" for check in checks)):
            raise ValueError("Incomplete or failed run")
        if len({case["id"] for case in cases}) != count:
            raise ValueError("Duplicate case identifier")
        parameters = metadata["parameters"]
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("Missing run parameters")
        canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf8")
        if not _digest_matches(canonical, metadata["parameters_sha256"]):
            raise ValueError("The parameter fingerprint differs")
        selected_file = output / "reviewer_parameters.json"
        if selected_file.exists() and _json(selected_file.read_bytes()) != parameters:
            raise ValueError("The selected parameters differ")
        summary = (output / "results.txt").read_bytes().replace(b"\r\n", b"\n")
        if not _digest_matches(summary, metadata["text_summary_sha256_lf"]):
            raise ValueError("The text summary differs")
        source_hashes = metadata["numerical_source_sha256_lf"]
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError("Missing numerical source fingerprints")
        for name, expected in source_hashes.items():
            relative = Path(name)
            allowed = name in ("test.py", "parameters.json") or (name.startswith("source/") and name.endswith(".py"))
            source = (project / relative).resolve(strict=True)
            if not allowed or relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(project):
                raise ValueError("Unexpected numerical source path")
            if not _digest_matches(source.read_bytes().replace(b"\r\n", b"\n"), expected):
                raise ValueError("The numerical source differs")
        csv_bytes = (output / "latest_results.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf8"))))
        if len(rows) != count:
            raise ValueError("The CSV case count differs")
        for row, case in zip(rows, cases):
            if set(row) != set(case) or any(row[key] != ("" if value is None else str(value)) for key, value in case.items()):
                raise ValueError("The CSV and JSON results differ")
        figure_hashes = metadata["figure_sha256"]
        if not isinstance(figure_hashes, dict) or set(figure_hashes) != set(FIGURES):
            raise ValueError("Missing figure fingerprints")
        files = {"data/latest_results.json": metadata_bytes, "data/latest_results.csv": csv_bytes,
                 "data/latest_parameters.json": (json.dumps(parameters, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf8")}
        for name in FIGURES:
            data = (output / name).read_bytes()
            _check_png(data)
            if not _digest_matches(data, figure_hashes[name]):
                raise ValueError("A figure fingerprint differs")
            files["images/" + name] = data
        if metadata["image_sha256"] != figure_hashes["slotted_sphere.png"]:
            raise ValueError("The sphere image fingerprint differs")
        readme = (output / "README.md").read_text(encoding="utf8")
        start, end = _bounds(readme)
        block = readme[start:end]
        if metadata["parameters_sha256"][:12] not in block or not _valid_sha(metadata["source_revision"]):
            raise ValueError("The README provenance differs")
        return block, files
    except PublicationError:
        raise
    except Exception:
        raise PublicationError("Publication needs a complete successful run with matching parameters, source, text, CSV and seven PNG fingerprints.") from None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "GitHub redirects are refused", headers, response)


def _transport(token):
    opener = build_opener(_NoRedirects())

    def request(method, path, payload=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Unexpected API path")
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf8") if payload is not None else None
        headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "BraudeauRichardsFEM-article", "Content-Type": "application/json"}
        query = Request(API_ROOT + path, data=body, headers=headers, method=method)
        with opener.open(query, timeout=45) as response:
            return _json(response.read())
    return request


def _call(transport, method, path, payload=None):
    try:
        response = transport(method, path, payload)
    except HTTPError as error:
        status = error.code if type(error.code) is int else 0
        if status in (409, 422):
            message = f"GitHub refused the update (HTTP {status}); the branch may have changed. Publish again without forcing it."
        elif status in (401, 403):
            message = f"GitHub refused the request (HTTP {status}). Check the owner's token and its Contents write permission."
        else:
            message = f"GitHub refused the request (HTTP {status}). No server response or credentials are displayed."
        raise PublicationError(message) from None
    except Exception:
        raise PublicationError("The GitHub request failed. No credentials or server response are displayed; the numerical results are still available.") from None
    if not isinstance(response, dict):
        raise PublicationError("GitHub returned an unexpected response.")
    return response


def _response_sha(response):
    value = response.get("sha")
    if not _valid_sha(value):
        raise PublicationError("GitHub returned an unexpected object identifier.")
    return value


def _head(transport):
    response = _call(transport, "GET", REPO_PATH + "/git/ref/heads/" + BRANCH)
    return _response_sha(response.get("object", {}))


def _blob_sha(data, length):
    content = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(content).hexdigest() if length == 40 else hashlib.sha256(content).hexdigest()


def publish_results(project_root, output_dir, token, transport=None):
    """Update eleven result files in one non-forced commit on the owner's main branch."""
    block, files = _prepare(project_root, output_dir)
    if not isinstance(token, str) or not token.strip() or any(character in token for character in "\r\n\0"):
        raise PublicationError("A valid owner token is required for GitHub publication.")
    send = transport if transport is not None else _transport(token.strip())
    user = _call(send, "GET", "/user")
    if user.get("login") != OWNER or user.get("type") != "User" or type(user.get("id")) is not int or user["id"] <= 0:
        raise PublicationError("Only the repository owner moalhadiri can publish these results.")
    repo = _call(send, "GET", REPO_PATH)
    if repo.get("full_name") != OWNER + "/" + REPOSITORY or repo.get("permissions", {}).get("push") is not True:
        raise PublicationError("The owner token needs Contents write permission for BraudeauRichardsFEM-article.")
    head = _head(send)
    previous = _call(send, "GET", REPO_PATH + "/git/commits/" + head)
    base_tree = _response_sha(previous.get("tree", {}))
    tree = _call(send, "GET", REPO_PATH + "/git/trees/" + base_tree + "?recursive=1")
    if tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise PublicationError("The current GitHub tree could not be read completely.")
    entries = {item.get("path"): item for item in tree["tree"]}
    readme_entry = entries.get("README.md", {})
    if readme_entry.get("type") != "blob" or not _valid_sha(readme_entry.get("sha")):
        raise PublicationError("The current GitHub README is missing.")
    readme_object = _call(send, "GET", REPO_PATH + "/git/blobs/" + readme_entry["sha"])
    try:
        if readme_object.get("encoding") != "base64":
            raise ValueError("Unexpected README encoding")
        remote_bytes = base64.b64decode("".join(readme_object["content"].split()), validate=True)
        if _blob_sha(remote_bytes, len(head)) != readme_entry["sha"]:
            raise ValueError("Unexpected README blob")
        remote_readme = remote_bytes.decode("utf8")
    except Exception:
        raise PublicationError("The current GitHub README could not be verified.") from None
    start, end = _bounds(remote_readme)
    files["README.md"] = (remote_readme[:start] + block + remote_readme[end:]).encode("utf8")
    changed = {name: data for name, data in files.items() if entries.get(name, {}).get("sha") != _blob_sha(data, len(head))
               or entries.get(name, {}).get("mode") != "100644" or entries.get(name, {}).get("type") != "blob"}
    if not changed:
        if _head(send) != head:
            raise PublicationError("The branch changed during publication. Publish again without forcing it.")
        return dict(status="unchanged", commit=head, url=f"https://github.com/{OWNER}/{REPOSITORY}/blob/{head}/README.md#computed-results",
                    commit_url=f"https://github.com/{OWNER}/{REPOSITORY}/commit/{head}", files=[])
    updates = []
    for name, data in changed.items():
        response = _call(send, "POST", REPO_PATH + "/git/blobs", dict(content=base64.b64encode(data).decode("ascii"), encoding="base64"))
        sha = _response_sha(response)
        if sha != _blob_sha(data, len(head)):
            raise PublicationError("GitHub returned an unexpected result blob.")
        updates.append(dict(path=name, mode="100644", type="blob", sha=sha))
    created_tree = _call(send, "POST", REPO_PATH + "/git/trees", dict(base_tree=base_tree, tree=updates))
    name = user.get("name") if isinstance(user.get("name"), str) and user["name"].strip() else OWNER
    identity = dict(name=" ".join(name.split()), email=f"{user['id']}+{OWNER}@users.noreply.github.com")
    created_commit = _call(send, "POST", REPO_PATH + "/git/commits",
                           dict(message="Update computed example results", tree=_response_sha(created_tree), parents=[head], author=identity, committer=identity))
    commit = _response_sha(created_commit)
    if _head(send) != head:
        raise PublicationError("The branch changed during publication. Publish again without forcing it.")
    updated = _call(send, "PATCH", REPO_PATH + "/git/refs/heads/" + BRANCH, dict(sha=commit, force=False))
    if _response_sha(updated.get("object", {})) != commit:
        raise PublicationError("GitHub did not confirm the requested branch update.")
    return dict(status="published", commit=commit, url=f"https://github.com/{OWNER}/{REPOSITORY}/blob/{commit}/README.md#computed-results",
                commit_url=f"https://github.com/{OWNER}/{REPOSITORY}/commit/{commit}", files=list(changed))


def publish_from_colab(project, output, enabled=True, get_secret=None):
    """Publish for the owner when a Colab secret is available; visitor runs stay local."""
    if not enabled:
        return dict(status="skipped", message="GitHub publication is disabled for this run.")
    instructions = ("GitHub publication skipped. Owner: create a fine-grained GitHub token for moalhadiri/BraudeauRichardsFEM-article "
                    "with Contents read and write, save it in Colab Secrets as BRAUDEAU_GITHUB_TOKEN, and allow this notebook access. "
                    "Visitors can ignore this step; the calculations, figures and results ZIP remain available.")
    try:
        if get_secret is None:
            from google.colab import userdata
            get_secret = userdata.get
        token = get_secret(SECRET_NAME)
        if not isinstance(token, str) or not token.strip():
            raise ValueError("No owner secret")
    except Exception:
        print(instructions)
        return dict(status="skipped", message=instructions)
    try:
        result = publish_results(project, output, token)
    except PublicationError as error:
        message = str(error)
        if token in message:
            message = "GitHub publication failed. The numerical results remain available."
        print(message)
        return dict(status="failed", message=message)
    except Exception:
        message = "GitHub publication failed. The numerical results remain available."
        print(message)
        return dict(status="failed", message=message)
    finally:
        token = None
    print(("GitHub results updated: " if result["status"] == "published" else "GitHub already has these results: ") + result["url"])
    return result
