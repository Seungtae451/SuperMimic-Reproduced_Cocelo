"""Audit BONES-SEED metadata before downloading or retargeting SOMA motions.

The paper's exact keyword list and 153 physically rejected clip IDs are not
published. Its name-filter approximation comes from this reproduction's
scripts/data/filter_bones_seed.py; V10 physics filtering remains a later step.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HAND_CATEGORIES = {'Object Manipulation', 'Consuming', 'Household'}
HAND_ACTION = re.compile(
    r'\b(?:grab|grasp|grip|hold|holding|carry|carrying|pick(?:ing)?[ _-]?up|'
    r'lift(?:ing)?|throw(?:ing)?|catch(?:ing)?|place|put[ _-]?down|'
    r'open(?:ing)?|close|closing|press(?:ing)?|pull(?:ing)?|push(?:ing)?|'
    r'drink(?:ing)?|eat(?:ing)?|smok(?:e|ing)|touch(?:ing)?)\b',
    re.IGNORECASE,
)
# The metadata's content_props=0 also covers pantomimed or unannotated objects.
# Name/description rules exclude motion concepts that require an external
# object, including kicks, props, vehicles and animal interactions. Keep
# generic object-free body gestures such as ball-change steps and self-touch.
UNSUPPORTED_OBJECT_MOTION = re.compile(
    r'(?:^|_)(?:lasso|leash|walk_the_dog|walk_big_dog|turn_walk_big_dog|'
    r'petting_dog|clean_up_after_dog|'
    r'painting|checking_time|horse_riding|throw_ball|pick_up|throw_trash|'
    r'kick_trash|catch_playing|mosquito|jumping_rope|feeding_birds|'
    r'read_book|read_newspaper|sewers_phone|eating_icecream|'
    r'passenger_hold_high_bus|funky_guitar|looking_by_binoculars|'
    r'pocket_searching|looking_in_the_mirror|street_avoid_car|'
    r'street_avoid_obstacle|street_watching_carcrash|'
    r'street_watching_traffic|street_passing_out_flyers|'
    r'street_taxi_wave|street_waiting_at_crossroad|high_jump|'
    r'take_a_sip|buckets|dance_vogue_handbag|rock_out|gunslinger|'
    r'greetings_hat)(?:_|$)',
    re.IGNORECASE,
)
UNSUPPORTED_OBJECT_DESCRIPTION = re.compile(
    r'\b(?:binoculars?|handbags?|bottles?|canvas(?:es)?|newspapers?|'
    r'umbrellas?|guitars?|laptops?|keyboards?|cigarettes?|'
    r'hats?|suitcases?|screwdrivers?|'
    r'leash(?:es)?|lassos?|flyers?|mirrors?|trash|icecream|'
    r'\bphones?|\bguns?|\bswords?|\bropes?)\b|'
    r'\b(?:moving pole|jump over (?:a |the )?pole|watch on (?:their|his|her) forearm|'
    r'leaning against a wall|throwing a ball|holding a ball|passing a ball|'
    r'(?:pick(?:ing)? up|collect(?:ing)?|drop(?:ping)?|grab(?:bing)?) '
    r'(?:an? |the )?(?:item|object|something)|(?:item|object) from the ground|'
    r'fixing (?:a |the |his |her )?tie|putting on (?:a |the )?jacket|'
    r'(?:item|object) in (?:left|right|both) hands?|'
    r'(?:fixing|shaking|holding|taking|take) (?:an? |the )?(?:item|object)|'
    r'(?:step(?:s|ping)? on (?:an? |the )?object)|'
    r'picking something up(?: from the floor)?)\b',
    re.IGNORECASE,
)
NATURAL_OBJECT_INTERACTION = re.compile(
    r'\b(?:pick(?:s|ing)? up (?:an? |the )?(?:item|object|something)|'
    r'reach(?:es|ing)? (?:down )?(?:for|towards?) (?:an? |the )?(?:item|object)|'
    r'(?:hold(?:s|ing)?|carr(?:y|ies|ying)|grab(?:s|bing)?) (?:an? |the )?'
    r'(?:item|object))\b',
    re.IGNORECASE,
)
COLUMNS = [
    'move_name', 'move_duration_frames', 'move_soma_uniform_path',
    'category', 'content_props', 'content_type_of_movement',
    'content_short_description', 'content_technical_description',
    'content_natural_desc_1', 'content_natural_desc_2', 'content_natural_desc_3',
]


def paper_name_terms(source: Path) -> list[str]:
    """Read the existing reproduction's published-paper approximation without importing tyro."""
    tree = ast.parse(source.read_text())
    values: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {'SONIC_KEYWORDS', 'PAPER_EXTRA'}:
                values[target.id] = ast.literal_eval(node.value)
    if set(values) != {'SONIC_KEYWORDS', 'PAPER_EXTRA'}:
        raise ValueError(f'Cannot read name-filter terms from {source}')
    return sorted(set(values['SONIC_KEYWORDS'] + values['PAPER_EXTRA']))


def hand_object_reason(row: dict) -> str | None:
    name = str(row['move_name'] or '').strip().lower()
    match = UNSUPPORTED_OBJECT_MOTION.search(name)
    if match:
        return f'unsupported_object_motion:{match.group().strip("_")}'
    description = ' '.join(str(row[k] or '') for k in (
        'content_short_description', 'content_technical_description'))
    match = UNSUPPORTED_OBJECT_DESCRIPTION.search(description)
    if match:
        return f'unsupported_object_description:{match.group().lower()}'
    # Natural descriptions sometimes contain detail omitted by technical fields.
    # The fourth is frequently unrelated to the motion, so use the first three
    # only for explicit interaction phrases rather than isolated object nouns.
    natural = ' '.join(str(row[k] or '') for k in (
        'content_natural_desc_1', 'content_natural_desc_2', 'content_natural_desc_3'))
    match = NATURAL_OBJECT_INTERACTION.search(natural)
    if match:
        return f'unsupported_object_natural:{match.group().lower()}'
    props = str(row['content_props'] or '').strip().lower()
    # A flat-ground, prop-free V10 simulation cannot reproduce the object's support/contact.
    if props not in {'', '0', '0.0', 'none'}:
        return 'prop_required'
    category = str(row['category'] or '')
    if category in HAND_CATEGORIES:
        return f'hand_category:{category}'
    movement = str(row['content_type_of_movement'] or '').strip().lower()
    if movement == 'object manipulation':
        return 'hand_movement:object_manipulation'
    if category == 'Object Interaction':
        description = ' '.join(str(row[k] or '') for k in (
            'move_name', 'content_short_description', 'content_technical_description'))
        if HAND_ACTION.search(description.replace('_', ' ')):
            return 'hand_action:Object Interaction'
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path,
                        default=root/'data/v10/downloads/metadata/seed_metadata_v004.parquet')
    parser.add_argument('--output-dir', type=Path, default=root/'data/v10/manifests')
    args = parser.parse_args()
    import pyarrow.parquet as pq

    metadata = args.metadata.resolve()
    if not metadata.is_file():
        parser.error(f'Metadata file not found: {metadata}')
    source = root/'scripts/data/filter_bones_seed.py'
    terms = paper_name_terms(source)
    rows = pq.read_table(metadata, columns=COLUMNS).to_pylist()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output/'bones_seed_prefilter.csv'
    tmp_path = output/'.bones_seed_prefilter.csv.tmp'
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    try:
        with tmp_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['move_name', 'soma_uniform_path', 'frames', 'status',
                             'reason', 'category', 'content_props'])
            for row in rows:
                name = str(row['move_name'])
                matched = next((term for term in terms if term in name.lower()), None)
                hand = hand_object_reason(row)
                if matched:
                    status, reason = 'exclude_paper_name', f'name:{matched}'
                elif hand:
                    status, reason = 'exclude_hand_object', hand
                else:
                    status, reason = 'candidate', ''
                status_counts[status] += 1
                if reason:
                    reason_counts[reason] += 1
                writer.writerow([name, row['move_soma_uniform_path'],
                                 row['move_duration_frames'], status, reason,
                                 row['category'], row['content_props']])
        os.replace(tmp_path, csv_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    download_manifest = root/'data/v10/downloads/download_manifest.json'
    revision = json.loads(download_manifest.read_text()).get('revision') if download_manifest.exists() else None
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_dataset': 'bones-studio/seed',
        'source_revision': revision,
        'source_parquet': str(metadata),
        'source_parquet_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'paper_name_rule_source': str(source),
        'paper_name_filter_is_approximate': True,
        'paper_reported_after_name': 129667,
        'paper_reported_after_g1_feasibility': 129514,
        'g1_feasibility_done': False,
        'v10_feasibility_done': False,
        'object_rule': 'exclude any motion with annotated props, an object-dependent category, or explicit person-object interaction in its motion name or descriptions, including non-hand interactions',
        'counts': dict(status_counts),
        'reason_counts': dict(reason_counts),
    }
    (output/'bones_seed_prefilter_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['counts'], indent=2))
    print(f'Output: {csv_path}')


if __name__ == '__main__':
    main()
