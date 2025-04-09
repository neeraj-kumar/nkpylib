"""Some tools to detect bpm in music.

Currently uses Essentia: https://essentia.upf.edu/tutorial_rhythm_beatdetection.html
"""

import json

from argparse import ArgumentParser

import essentia.standard as es # type: ignore


if __name__ == '__main__':
    parser = ArgumentParser(description='Beat Per Minute (bpm) finder')
    parser.add_argument('input_audio', help='Path to input audio file')
    parser.add_argument('-a', '--output_audio', help='Output audio file (with beeps)')
    parser.add_argument('-j', '--output_json', help='Output json file')
    args = parser.parse_args()
    audio = es.MonoLoader(filename=args.input_audio)()
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    print(f'Got bpm {bpm} with confidence {beats_confidence}')
    if args.output_audio:
        marker = es.AudioOnsetsMarker(onsets=beats, type='beep')
        marked_audio = marker(audio)
        print(f'Writing audio output to {args.output_audio}')
        es.MonoWriter(filename=args.output_audio)(marked_audio)
    if args.output_json:
        data = dict(bpm=bpm, beats=[float(b) for b in beats], beats_confidence=beats_confidence)
        print(f'Writing json output to {args.output_json}')
        with open(args.output_json, 'w') as f:
            json.dump(data, f, indent=2)
