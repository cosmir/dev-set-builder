#!/usr/bin/env python
'''Namespace conversions into openmic schema'''

import jams


@jams.nsconvert._conversion('tag_openmic25', 'tag_medleydb_instruments')
def medley_to_openmic(annotation):

    data = annotation.pop_data()

    for obs in data:
        if obs.value in MEDLEY_MAP:
            annotation.append(time=obs.time,
                              duration=obs.duration,
                              confidence=obs.confidence,
                              value=MEDLEY_MAP[obs.value])

    return annotation


@jams.nsconvert._conversion('tag_openmic25', 'tag_irmas_instruments')
def irmas_to_openmic(annotation):

    data = annotation.pop_data()

    for obs in data:
        if obs.value in IRMAS_MAP:
            annotation.append(time=obs.time,
                              duration=obs.duration,
                              confidence=obs.confidence,
                              value=IRMAS_MAP[obs.value])

    return annotation


@jams.nsconvert._conversion('tag_openmic25', 'tag_audioset_instruments')
def audioset_to_openmic(annotation):

    data = annotation.pop_data()

    for obs in data:
        if obs.value in AUDIOSET_MAP:
            annotation.append(time=obs.time,
                              duration=obs.duration,
                              confidence=obs.confidence,
                              value=AUDIOSET_MAP[obs.value])

    return annotation


MEDLEY_MAP = {'accordion': 'accordion',
              'acoustic guitar': 'guitar',
              'alto saxophone': 'saxophone',
              'bamboo flute': 'flute',
              'banjo': 'banjo',
              'baritone saxophone': 'saxophone',
              'bass clarinet': 'clarinet',
              'bass drum': 'drums',
              'cello': 'cello',
              'cello section': 'cello',
              'clarinet': 'clarinet',
              'clarinet section': 'clarinet',
              'clean electric guitar': 'guitar',
              'cymbal': 'cymbals',
              'distorted electric guitar': 'guitar',
              'double bass': 'bass',
              'drum machine': 'drums',
              'drum set': 'drums',
              'electric bass': 'bass',
              'electric guitar': 'guitar',
              'electric piano': 'piano',
              'female rapper': 'voice',
              'female singer': 'voice',
              'female speaker': 'voice',
              'flute': 'flute',
              'flute section': 'flute',
              'glockenspiel': 'mallet_percussion',
              'harmonica': 'harmonica',
              'harp': 'harp',
              'kick drum': 'drums',
              'lap steel guitar': 'guitar',
              'male rapper': 'voice',
              'male singer': 'voice',
              'male speaker': 'voice',
              'mandolin': 'mandolin',
              'oboe': 'oboe',
              'piano': 'piano',
              'snare drum': 'drums',
              'soprano saxophone': 'saxophone',
              'synthesizer': 'synthesizer',
              'tack piano': 'piano',
              'tenor saxophone': 'saxophone',
              'toms': 'drums',
              'trombone': 'trombone',
              'trombone section': 'trombone',
              'trumpet': 'trumpet',
              'trumpet section': 'trumpet',
              'tuba': 'tuba',
              'upright bass': 'bass',
              'vibraphone': 'mallet_percussion',
              'violin': 'violin',
              'violin section': 'violin',
              'vocalists': 'voice'}

IRMAS_MAP = {'cello': 'cello',
             'clarinet': 'clarinet',
             'drums': 'drums',
             'flute': 'flute',
             'guitar (acoustic)': 'guitar',
             'guitar (electric)': 'guitar',
             'organ': 'organ',
             'piano': 'piano',
             'saxophone': 'saxophone',
             'trumpet': 'trumpet',
             'violin': 'violin',
             'voice': 'voice'}

AUDIOSET_MAP = {'Accordion': 'accordion',
                'Acoustic guitar': 'guitar',
                'Alto saxophone': 'saxophone',
                'Bagpipes': 'bagpipes',
                'Banjo': 'banjo',
                'Bass drum': 'drums',
                'Bass guitar': 'bass',
                'Cello': 'cello',
                'Choir': 'voice',
                'Clarinet': 'clarinet',
                'Crash cymbal': 'cymbals',
                'Cymbal': 'cymbals',
                'Double bass': 'bass',
                'Drum': 'drums',
                'Drum kit': 'drums',
                'Drum machine': 'drums',
                'Drum roll': 'drums',
                'Electric guitar': 'guitar',
                'Electric piano': 'piano',
                'Electronic organ': 'organ',
                'Flute': 'flute',
                'Glockenspiel': 'mallet_percussion',
                'Guitar': 'guitar',
                'Hammond organ': 'organ',
                'Harmonica': 'harmonica',
                'Harp': 'harp',
                'Hi-hat': 'cymbals',
                'Mallet percussion': 'mallet_percussion',
                'Mandolin': 'mandolin',
                'Marimba, xylophone': 'mallet_percussion',
                'Oboe': 'oboe',
                'Organ': 'organ',
                'Piano': 'piano',
                'Rhodes piano': 'piano',
                'Rimshot': 'drums',
                'Saxophone': 'saxophone',
                'Singing': 'voice',
                'Yodeling': 'voice',
                'Male singing': 'voice',
                'Female singing': 'voice',
                'Child singing': 'voice',
                'Snare drum': 'drums',
                'Soprano saxophone': 'saxophone',
                'Steel guitar, slide guitar': 'guitar',
                'Synthesizer': 'synthesizer',
                'Tapping (guitar technique)': 'guitar',
                'Trombone': 'trombone',
                'Trumpet': 'trumpet',
                'Ukulele': 'ukulele',
                'Vibraphone': 'mallet_percussion',
                'Violin, fiddle': 'violin'}
