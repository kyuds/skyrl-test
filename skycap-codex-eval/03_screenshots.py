"""Screenshots of an eval run in skycap-viewer, for a review: the run's groups and rollouts, and the Path, Tree and
Calls tabs of a few rollouts (every forked one, up to --forks, plus the longest clean accepted one and a clean
failed one).

Runs anywhere a headless Chromium can reach the viewer: on your Mac through gcp/tunnel.sh (the eval viewer is
forwarded to :8766). Writes PNGs and an index (screenshots/README.md) to runs/<run>/screenshots/.

    uv run --no-project --isolated --with playwright python -m playwright install chromium     # once
    uv run --no-project --isolated --with playwright --with httpx python skyrl-test/skycap-codex-eval/03_screenshots.py <run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

KIT = Path(__file__).resolve().parent
TABS = ("Path", "Tree", "Forks", "Calls")


def pick(rows: list[dict], forks: int) -> list[tuple[str, dict]]:
    """(why, row) for the rollouts worth a picture."""
    flags = lambda r: (r.get("summary") or {}).get("flags") or []
    calls = lambda r: (r.get("capture") or {}).get("exchange_count") or 0
    reward = lambda r: (r.get("annotations") or {}).get("reward")
    chosen: list[tuple[str, dict]] = [("forked", r) for r in rows if "forked" in flags(r)][:forks]
    clean = [r for r in rows if not {"forked", "unbridged"} & set(flags(r))]
    accepted = sorted((r for r in clean if reward(r) == 1), key=calls, reverse=True)
    failed = sorted((r for r in clean if reward(r) == 0), key=calls, reverse=True)
    if accepted:
        chosen.append(("clean, accepted, most calls", accepted[0]))
    if failed:
        chosen.append(("clean, not accepted", failed[0]))
    if not accepted and not failed and clean:
        chosen.append(("clean, most calls", max(clean, key=calls)))
    return chosen


def settle(page: Page) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1200)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="the run's name (its record directory under records/)")
    parser.add_argument("--viewer", default="http://127.0.0.1:8766")
    parser.add_argument("--out", type=Path, help="default: runs/<run>/screenshots")
    parser.add_argument("--forks", type=int, default=3, help="forked rollouts to picture, at most")
    args = parser.parse_args()
    viewer = args.viewer.rstrip("/")
    out = args.out or KIT / "runs" / args.run / "screenshots"
    out.mkdir(parents=True, exist_ok=True)

    runs = httpx.get(f"{viewer}/api/v1/runs", params={"refresh": "true"}, timeout=30).json()["data"]
    ids = [r["id"] for r in runs]
    if args.run not in ids:
        raise SystemExit(f"the viewer at {viewer} has no run {args.run!r}; it has {ids}")
    rows = httpx.get(f"{viewer}/api/v1/trajectories", params={"run_id": args.run, "limit": 1000}, timeout=60).json()[
        "data"
    ]
    chosen = pick(rows, args.forks)
    index = [f"# skycap-viewer: {args.run}", ""]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1050}, device_scale_factor=2)
        for name, route, caption in (
            ("01-groups.png", f"#/run/{args.run}/groups", "Groups: every task's rollouts, coloured by reward"),
            ("02-rollouts.png", f"#/run/{args.run}/rollouts", "Rollouts, with the health band (forked, unbridged, truncated)"),
        ):  # fmt: skip
            page.goto(f"{viewer}/{route}")
            settle(page)
            page.screenshot(path=out / name, full_page=True)
            index += [f"## {caption}", "", f"![{caption}]({name})", ""]
        for number, (why, row) in enumerate(chosen, 3):
            task = (row.get("meta") or {}).get("task_id") or row["id"]
            page.goto(f"{viewer}/#/run/{args.run}/rollouts/{row['id']}")
            settle(page)
            index += [f"## {why}: `{row['id']}` ({task}, reward {(row.get('annotations') or {}).get('reward')})", ""]
            for tab in TABS:
                if tab == "Forks" and why != "forked":
                    continue
                button = page.locator(".tabs button", has_text=tab)
                if button.count() == 0:
                    print(f"note: no {tab} tab for {row['id']}")
                    continue
                button.first.click()
                settle(page)
                name = f"{number:02d}-{row['id']}-{tab.lower()}.png"
                page.screenshot(path=out / name)
                index += [f"{tab}:", "", f"![{tab}]({name})", ""]
            print(f"{why}: {row['id']}")
        browser.close()
    (out / "README.md").write_text("\n".join(index))
    print(f"Screenshots: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
