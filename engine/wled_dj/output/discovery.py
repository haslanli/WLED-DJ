"""Find WLED controllers on the local network (they advertise _wled._tcp over mDNS)."""
from __future__ import annotations

import socket
import time

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

SERVICE = "_wled._tcp.local."


def discover(timeout: float = 3.0) -> list[tuple[str, str]]:
    names: set[str] = set()

    class Listener(ServiceListener):
        def add_service(self, zc, type_, name):
            names.add(name)

        def update_service(self, zc, type_, name):
            names.add(name)

        def remove_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE, Listener())
        time.sleep(timeout)
        found = set()
        for name in sorted(names):
            info = zc.get_service_info(SERVICE, name, timeout=1000)
            if info is None:
                continue
            for address in info.addresses:
                if len(address) == 4:
                    found.add((name[:-len("." + SERVICE)] if name.endswith("." + SERVICE) else name,
                               socket.inet_ntoa(address)))
        return sorted(found)
    finally:
        zc.close()
