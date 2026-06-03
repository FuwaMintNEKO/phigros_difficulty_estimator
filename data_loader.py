import os
import json
import csv


def load_difficulty_tsv(tsv_path):
    song_difficulties = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if not row or len(row) < 2:
                continue
            song_id = row[0].strip()
            diffs = {}
            level_names = ['EZ', 'HD', 'IN']
            for i, val in enumerate(row[1:]):
                level = level_names[i] if i < len(level_names) else 'AT'
                try:
                    diffs[level] = float(val)
                except ValueError:
                    pass
            song_difficulties[song_id] = diffs
    return song_difficulties


def find_chart_files(chart_dir):
    chart_files = {}
    for folder_name in os.listdir(chart_dir):
        folder_path = os.path.join(chart_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        song_id = folder_name
        if song_id.endswith('.0'):
            song_id = song_id[:-2]

        levels = {}
        for fname in os.listdir(folder_path):
            if fname.endswith('.json'):
                level = fname.replace('.json', '')
                levels[level] = os.path.join(folder_path, fname)

        if levels:
            chart_files[folder_name] = {'song_id': song_id, 'levels': levels}
    return chart_files


def load_chart_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)