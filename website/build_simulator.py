"""Assemble simulator.html = head.html + models bundle + core.js + ui.js."""
import sys
from pathlib import Path

site = Path(__file__).resolve().parent
d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/loop")
b = (d / "models_bundle.json").read_text(encoding="utf-8")
src = site / "browser_sim"
t = (src / "head.html").read_text(encoding="utf-8") + f"const MODELS={b};\n" + (src / "core.js").read_text(encoding="utf-8") + (src / "ui.js").read_text(encoding="utf-8").replace("</script></body></html>", "") + "</script></body></html>"
output = Path(sys.argv[2]) if len(sys.argv) > 2 else site / "simulator.html"
output.write_text(t, encoding="utf-8")
print("simulator bytes", len(t))
