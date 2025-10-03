    import argparse, time, psutil, json, sys

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--pid", type=int, help="PID процесса для мониторинга")
        ap.add_argument("--process-name", help="Часть имени процесса для поиска (если PID не задан)")
        ap.add_argument("--interval", type=float, default=1.0)
        ap.add_argument("--out", required=True, help="CSV-файл для записи метрик")
        args = ap.parse_args()

        pid = args.pid
        if not pid and args.process_name:
            # find first process by name substring
            for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
                cmd = " ".join(p.info.get("cmdline") or [])
                if args.process_name.lower() in (p.info.get("name","").lower() + " " + cmd.lower()):
                    pid = p.info["pid"]
                    break
        if not pid:
            print("PID не найден", file=sys.stderr)
            sys.exit(2)

        p = psutil.Process(pid)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("t_seconds,rss_mb,vms_mb,cpu_percent
")
            t0 = time.time()
            try:
                p.cpu_percent(None)
            except Exception:
                pass
            peak = 0.0
            while True:
                if not p.is_running():
                    break
                try:
                    rss = p.memory_info().rss / (1024*1024)
                    vms = p.memory_info().vms / (1024*1024)
                    cpu = p.cpu_percent(interval=args.interval)
                    f.write(f"{time.time()-t0:.2f},{rss:.2f},{vms:.2f},{cpu:.2f}\n")
                    f.flush()
                    if rss > peak: peak = rss
                except psutil.NoSuchProcess:
                    break
                except Exception:
                    time.sleep(args.interval)
        print(f"peak_rss_mb={peak:.2f}")

    if __name__ == "__main__":
        main()
