import paddleocr
import os
import glob

d = os.path.join(os.path.dirname(paddleocr.__file__), 'ppocr', 'utils', 'dict')
print('Dict dir:', d)

EXTENDED_SPECIAL_CHARS = [
    '?', ',', '.', '!', '~', '…', '-', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '“', '”', '‘', '’',
    '¿', '¡', '«', '»', '‹', '›', '–', '—', '―', '‽', '⸘',
    '。', '、', '・', '！', '？', '：', '；',
    '「', '」', '『', '』', '【', '】', '《', '》', '〈', '〉', '〔', '〕', '〛', '〚',
    '°', '©', '®', '™', '×', '÷', '±', '²', '³', '½', '¼', '¾', '†', '‡', '§', '¶',
    '¢', '€', '£', '¥', '₩',
    '•', '◦', '⁃', '❤', '♡', '♥', '★', '☆', '♪', '♫',
    '✓', '✔', '✕', '✖', '✗', '✘',
    '......', '...', '!!', '?!', '!?', '⁉', '⁈', '⁇', '⁉️'
]

dicts = glob.glob(os.path.join(d, '*dict.txt'))
for df in dicts:
    try:
        with open(df, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f)
        
        added = False
        with open(df, 'a', encoding='utf-8') as f:
            for p in EXTENDED_SPECIAL_CHARS:
                for char in p:
                    if char not in existing:
                        f.write('\n' + char)
                        existing.add(char)
                        added = True
        if added:
            print(f"Updated {df}")
    except Exception as e:
        print(f"Failed {df}: {e}")
print('Done.')
