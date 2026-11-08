import socket
import select
import time

ports = [4242, 6969, 8087, 8088, 4545]
socks = []

for p in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", p))
    socks.append((s, p))
    print(f"Listening UDP port {p}")

tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp.bind(("0.0.0.0", 4242))
tcp.listen(1)
tcp.setblocking(False)
print("Listening TCP port 4242")
print("\nWaiting for CoT data...")

while True:
    readable, _, _ = select.select([s for s,p in socks] + [tcp], [], [], 1)
    for sock in readable:
        if sock == tcp:
            conn, addr = sock.accept()
            conn.setblocking(True)
            data = conn.recv(4096)
            print(f"\nTCP 4242 from {addr}:")
            print(data.decode("utf-8", errors="replace")[:500])
            conn.close()
        else:
            port = [p for s,p in socks if s == sock][0]
            data, addr = sock.recvfrom(4096)
            print(f"\nUDP {port} from {addr}:")
            print(data.decode("utf-8", errors="replace")[:500])