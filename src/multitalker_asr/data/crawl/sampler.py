"""
Stratified random sampler over (platform, tag) cells of the crawl DB.

Guarantees every non-empty (platform, tag) cell is represented (floor), then
distributes the remaining budget proportionally to cell population. Sampling
within a cell is uniform random over a stable created_at ordering, so it never
downloads the full population. Deduplicates items shared across cells.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .client import CrawlAPIClient


@dataclass
class SampleCell:
    platform: str
    tag_id: int
    count: int = 0
    quota: int = 0


@dataclass
class SampleResult:
    tasks: List[Dict] = field(default_factory=list)
    realized: Dict[str, int] = field(default_factory=dict)


class StratifiedSampler:
    def __init__(self, page_limit: int = 1000, floor: int = 1):
        self._page_limit = page_limit
        self._floor = floor

    def sample(
        self,
        client: CrawlAPIClient,
        platforms: List[str],
        tag_ids: List[int],
        total: int,
        seed: int = 42,
        status: str = "completed",
    ) -> SampleResult:
        rng = random.Random(seed)
        cells = self._build_cells(client, platforms, tag_ids, status)
        if not cells:
            logger.warning("No non-empty (platform, tag) cells found")
            return SampleResult()

        self._allocate(cells, total)

        seen: set = set()
        tasks: List[Dict] = []
        realized: Dict[str, int] = {}
        for cell in cells:
            if cell.quota <= 0:
                continue
            picked = self._sample_cell(client, cell, rng, status, seen)
            tasks.extend(picked)
            key = f"{cell.platform}:{cell.tag_id}"
            realized[key] = len(picked)

        logger.info(f"Sampled {len(tasks)} unique tasks across {len(cells)} cells")
        return SampleResult(tasks=tasks, realized=realized)

    def _build_cells(
        self,
        client: CrawlAPIClient,
        platforms: List[str],
        tag_ids: List[int],
        status: str,
    ) -> List[SampleCell]:
        cells: List[SampleCell] = []
        for platform in platforms:
            for tag_id in tag_ids:
                count = client.count(platform=platform, tag_ids=[tag_id], status=status)
                if count > 0:
                    cells.append(SampleCell(platform=platform, tag_id=tag_id, count=count))
        return cells

    def _allocate(self, cells: List[SampleCell], total: int) -> None:
        n = len(cells)
        for cell in cells:
            cell.quota = min(cell.count, self._floor)

        remaining = total - sum(c.quota for c in cells)
        if remaining <= 0:
            return

        headroom = [(c, c.count - c.quota) for c in cells]
        pool = sum(h for _, h in headroom)
        if pool <= 0:
            return

        for cell, head in headroom:
            add = int(round(remaining * head / pool))
            cell.quota = min(cell.count, cell.quota + add)

        drift = total - sum(c.quota for c in cells)
        if drift > 0:
            for cell in sorted(cells, key=lambda c: c.count - c.quota, reverse=True):
                if drift <= 0:
                    break
                room = cell.count - cell.quota
                take = min(room, drift)
                cell.quota += take
                drift -= take

    def _sample_cell(
        self,
        client: CrawlAPIClient,
        cell: SampleCell,
        rng: random.Random,
        status: str,
        seen: set,
    ) -> List[Dict]:
        quota = min(cell.quota, cell.count)
        indices = sorted(rng.sample(range(cell.count), quota))
        by_page: Dict[int, List[int]] = {}
        for idx in indices:
            page = idx // self._page_limit + 1
            by_page.setdefault(page, []).append(idx % self._page_limit)

        picked: List[Dict] = []
        for page, offsets in by_page.items():
            tasks = client.get_page(
                platform=cell.platform,
                tag_ids=[cell.tag_id],
                status=status,
                page=page,
                limit=self._page_limit,
            )
            for off in offsets:
                if off >= len(tasks):
                    continue
                task = tasks[off]
                tid = task.get("task_id")
                if tid in seen:
                    continue
                seen.add(tid)
                picked.append(task)
        return picked
