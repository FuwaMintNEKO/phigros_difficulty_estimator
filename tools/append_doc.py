import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('v8_9_changelog.md', 'r', encoding='utf-8') as f:
    content = f.read()
with open('Phigros谱面定数估计.md', 'a', encoding='utf-8') as f:
    f.write('\n\n---\n\n' + content)
print("Appended v8.9 changelog to main document")