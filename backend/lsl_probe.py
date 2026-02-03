import json
import time
from dataclasses import dataclass

from pylsl import resolve_streams


@dataclass
class StreamInfoLite:
    name: str
    type: str
    channel_count: int
    nominal_srate: float
    channel_format: str
    source_id: str


def probe(timeout: float = 2.0) -> dict:
    t0 = time.time()
    streams = resolve_streams(wait_time=timeout)
    out = {
        "timestamp": time.time(),
        "elapsed_s": time.time() - t0,
        "count": len(streams),
        "streams": [],
    }

    for si in streams:
        info = StreamInfoLite(
            name=si.name(),
            type=si.type(),
            channel_count=si.channel_count(),
            nominal_srate=si.nominal_srate(),
            channel_format=str(si.channel_format()),
            source_id=si.source_id(),
        )
        out["streams"].append({"info": info.__dict__, "xml": si.as_xml()})

    return out


def main():
    print(json.dumps(probe(), indent=2))


if __name__ == "__main__":
    main()
