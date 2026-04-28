from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DROP_KEYWORDS = (
    "hand",
    "head",
    "face",
    "gaze",
    "eye",
    "eyes",
    "mirror",
)


def should_keep_action(action_type: str, drop_keywords: tuple[str, ...]) -> bool:
    action_type_lower = action_type.lower()
    return not any(keyword in action_type_lower for keyword in drop_keywords)


def filter_streams(streams: dict[str, Any]) -> dict[str, Any]:
    body_streams: dict[str, Any] = {}
    for stream_name, stream_data in streams.items():
        stream_name_lower = stream_name.lower()
        description = str(stream_data.get("description", "")).lower()
        if "body" in stream_name_lower or "body" in description:
            body_streams[stream_name] = stream_data
    return body_streams


def filter_actions(actions: dict[str, Any], drop_keywords: tuple[str, ...]) -> dict[str, Any]:
    filtered_actions: dict[str, Any] = {}
    for action_id, action_data in actions.items():
        action_type = str(action_data.get("type", ""))
        if should_keep_action(action_type, drop_keywords):
            filtered_actions[action_id] = {
                "type": action_type,
                "frame_intervals": action_data.get("frame_intervals", []),
            }
    return filtered_actions


def clean_dmd_json(input_path: Path, output_path: Path, drop_keywords: tuple[str, ...]) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    openlabel = payload.get("openlabel", {})
    cleaned_openlabel: dict[str, Any] = {}

    for key in ("metadata", "frame_intervals"):
        if key in openlabel:
            cleaned_openlabel[key] = openlabel[key]

    if "streams" in openlabel:
        cleaned_openlabel["streams"] = filter_streams(openlabel["streams"])

    if "actions" in openlabel:
        cleaned_openlabel["actions"] = filter_actions(openlabel["actions"], drop_keywords)

    cleaned_payload = {"openlabel": cleaned_openlabel}

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return cleaned_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only body-related annotations from a DMD OpenLABEL JSON file."
    )
    parser.add_argument("input_json", type=Path, help="Path to the original DMD annotation JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for the cleaned JSON. Defaults to <input>_body_only.json",
    )
    parser.add_argument(
        "--drop-keyword",
        action="append",
        default=list(DROP_KEYWORDS),
        help="Keyword used to drop non-body actions. Can be repeated.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input_json
    output_path: Path = args.output or input_path.with_name(f"{input_path.stem}_body_only.json")
    drop_keywords = tuple(keyword.lower() for keyword in args.drop_keyword)

    cleaned_payload = clean_dmd_json(input_path, output_path, drop_keywords)
    actions = cleaned_payload.get("openlabel", {}).get("actions", {})
    streams = cleaned_payload.get("openlabel", {}).get("streams", {})

    print(f"Written: {output_path}")
    print(f"Kept actions: {len(actions)}")
    print(f"Kept streams: {', '.join(streams.keys()) if streams else 'none'}")


if __name__ == "__main__":
    main()