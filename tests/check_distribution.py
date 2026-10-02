"""Check release contents without installing SNACK or its dependencies.

Run from the repository root: python tests/check_distribution.py dist/*
"""
import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent


def expected_payload():
    paths = list((ROOT / 'snack').glob('*.py'))
    datasets = list((ROOT / 'snack' / 'data').glob('*.csv'))
    if not datasets:
        raise ValueError('No source datasets found')
    paths.extend(datasets)
    paths.append(ROOT / 'snack' / 'data' / 'README.md')
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}


def archive_files(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            names = [member.filename for member in members]
            files = {member.filename: archive.read(member) for member in members}
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as archive:
            members = [member for member in archive.getmembers() if not member.isdir()]
            if any(not member.isfile() for member in members):
                raise ValueError('Source archives must contain only regular files and directories')
            roots = {PurePosixPath(member.name).parts[0] for member in members}
            if len(roots) != 1:
                raise ValueError('Source archive must have one root directory')
            names = [member.name.split('/', 1)[1] for member in members]
            files = {name: archive.extractfile(member).read()
                     for name, member in zip(names, members)}
    else:
        raise ValueError(f'Expected a wheel or .tar.gz source archive: {path}')
    if len(names) != len(files):
        raise ValueError('Archive contains duplicate file names')
    return files


def check_archive(path):
    files = archive_files(path)
    expected = expected_payload()
    payload = {name: content for name, content in files.items() if name.startswith('snack/')}
    missing = expected.keys() - payload.keys()
    extra = payload.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f'Package contents differ: missing={sorted(missing)}, extra={sorted(extra)}')
    for name, content in expected.items():
        if payload[name] != content:
            raise ValueError(f'Packaged content differs from source: {name}')

    metadata_paths = []
    for name in files.keys() - payload.keys():
        parts = PurePosixPath(name).parts
        if '..' in parts or name.startswith('/'):
            raise ValueError(f'Invalid archive path: {name}')
        if path.suffix == '.whl':
            if not parts[0].endswith('.dist-info'):
                raise ValueError(f'Unexpected wheel file: {name}')
            if parts[1:] == ('METADATA',):
                metadata_paths.append(name)
        else:
            if name == 'PKG-INFO':
                metadata_paths.append(name)
            if name in {'PKG-INFO', 'pyproject.toml', 'MANIFEST.in', 'README.md', 'setup.cfg'}:
                continue
            if len(parts) == 1 and name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
                continue
            if len(parts) == 2 and parts[0].endswith('.egg-info') and parts[1] in {
                    'PKG-INFO', 'SOURCES.txt', 'dependency_links.txt', 'requires.txt',
                    'top_level.txt', 'entry_points.txt'}:
                continue
            raise ValueError(f'Unexpected source archive file: {name}')
    if len(metadata_paths) != 1:
        raise ValueError('Expected exactly one package metadata file')
    metadata = BytesParser().parsebytes(files[metadata_paths[0]])
    if not metadata['Name'] or not metadata['Version'] or not metadata['Requires-Python']:
        raise ValueError('Missing package name, version, or Python requirement')
    dataset_count = sum(name.endswith('.csv') for name in payload)
    print(f'{path.name}: {len(payload)} package files, {dataset_count} datasets; '
          f'no images or tests ({path.stat().st_size / 1024:.1f} KiB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.archives:
        try:
            check_archive(path)
        except (OSError, ValueError, KeyError, IndexError, tarfile.TarError,
                zipfile.BadZipFile) as exc:
            parser.exit(1, f'{path}: {exc}\n')


if __name__ == '__main__':
    main()
