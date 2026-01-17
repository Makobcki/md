from __future__ import annotations


def _build_curriculum_weights(
    entries: list[dict],
    *,
    require_one_person: bool,
    prefer_solo: bool,
    exclude_multi: bool,
    solo_weight: float,
    non_solo_weight: float,
) -> list[float]:
    if solo_weight <= 0 or non_solo_weight <= 0:
        raise RuntimeError("curriculum weights must be positive.")
    exclude_tags = {
        "2girls",
        "2boys",
        "3girls",
        "3boys",
        "multiple_girls",
        "multiple_boys",
        "group",
        "crowd",
    }
    weights = []
    for entry in entries:
        tags_primary = [str(t).lower() for t in entry.get("tags_primary", [])]
        tags_gender = [str(t).lower() for t in entry.get("tags_gender", [])]
        tags = set(tags_primary + tags_gender)
        if exclude_multi and any(tag in tags for tag in exclude_tags):
            weights.append(0.0)
            continue
        has_one_person = ("1girl" in tags) or ("1boy" in tags)
        if require_one_person and not has_one_person:
            weights.append(0.0)
            continue
        if prefer_solo and "solo" in tags:
            weights.append(float(solo_weight))
        elif prefer_solo:
            weights.append(float(non_solo_weight))
        else:
            weights.append(1.0)
    if all(w == 0.0 for w in weights):
        raise RuntimeError("Curriculum sampler has no eligible entries; check tag filters.")
    return weights
