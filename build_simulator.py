"""Assemble simulator.html = head.html + models bundle + core.js + ui.js."""
import sys
d = sys.argv[1] if len(sys.argv) > 1 else "results/loop"
b = open(f"{d}/models_bundle.json").read(); s = "browser_sim/"
t = open(s + "head.html").read() + f"const MODELS={b};\n" + open(s + "core.js").read() + open(s + "ui.js").read().replace("</script></body></html>", "") + "</script></body></html>"
open(sys.argv[2], "w").write(t); print("simulator bytes", len(t))
