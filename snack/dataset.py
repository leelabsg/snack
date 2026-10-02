import csv
from importlib.resources import files
import re


def load_dataset(name: str) -> dict[str, list[str]]:
    """Return {"train": [...], "val": [...], "test": [...]} of SNACK strings.

    Supported dataset names:
        - community_small
        - ego
        - ego_small
        - enzymes
        - grid
        - grid_small
        - lobster
        - planar
        - sbm

    Raises:
        ValueError: Invalid name or malformed split metadata.
        FileNotFoundError: No bundled CSV exists for the requested name.
    """
    if not isinstance(name, str) or re.fullmatch(r'[a-z][a-z0-9_]*', name) is None:
        raise ValueError('Dataset name must be a lowercase stem, e.g. "planar"')

    resource = files(__package__).joinpath('data', f'{name}.csv')
    splits = {key: [] for key in ('train', 'val', 'test')}
    with resource.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'snack', 'split', 'split_index'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{name}.csv must contain snack, split and split_index columns')
        for row in reader:
            split = row['split']
            if split not in splits or row['snack'] is None or None in row:
                raise ValueError(f'{name}.csv line {reader.line_num}: invalid dataset row')
            try:
                index = int(row['split_index'])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{name}.csv line {reader.line_num}: invalid split_index') from exc
            splits[split].append((index, row['snack']))

    result = {}
    for split, rows in splits.items():
        rows.sort(key=lambda row: row[0])
        if [index for index, _ in rows] != list(range(len(rows))):
            raise ValueError(f'{name}.csv: {split} split_index must be unique and consecutive from 0')
        result[split] = [snack for _, snack in rows]
    return result
