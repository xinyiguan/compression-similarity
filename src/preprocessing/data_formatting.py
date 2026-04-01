import json
from typing import Dict, Any, List

import music21 as m21


def chord_transition_format(chord1_str: str, chord2_str: str) -> str:
    """
    Format the chord transition as a string.

    Args:
        chord1_str (str): The first chord.
        chord2_str (str): The second chord.

    Returns:
        str: A formatted string representing the chord transition.
    """
    chord1 = chord1_str.replace('♭', 'b').replace('♯', '#')
    chord2 = chord2_str.replace('♭', 'b').replace('♯', '#')
    pitch1 = chord1[:2] if chord1[1] in ['#', 'b', '-'] else chord1[0]
    pitch2 = chord2[:2] if chord2[1] in ['#', 'b', '-'] else chord2[0]
    quality1 = chord1[2:] if len(pitch1) == 2 else chord1[1:]
    quality2 = chord2[2:] if len(pitch2) == 2 else chord2[1:]
    if quality1 == '':
        quality1 = 'maj'
    if quality2 == '':
        quality2 = 'maj'

    interval = m21.interval.Interval(m21.pitch.Pitch(pitch1), m21.pitch.Pitch(pitch2)).semitones % 12
    return f'{interval}-{quality1}-{quality2}'


def convert_treebank_to_corpus_rep(treebank_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a treebank entry to CorpusRep format.
    
    Args:
        treebank_entry: A dictionary from treebank.json containing:
            - title: The piece title
            - chords: List of chord symbols
            - (optional) trees: The tree structure representing the harmonic analysis
    
    Returns:
        A dictionary in CorpusRep format containing:
            - pieceTitle: The piece title (with validation status prefix)
            - progression: Space-separated chord transitions
            - program: The harmonic structure as a nested function representation
        
        Returns None if the entry doesn't have a valid tree structure.
    
    Example:
        treebank_entry = {
            "title": "Red Clay",
            "chords": ["Cm7", "B♭m7", "D♭sus", "E♭sus", ...],
            "trees": [{"complete_constituent_tree": {...}}]
        }
        corpus_rep = convert_treebank_to_corpus_rep(treebank_entry)
    """
    trees = treebank_entry.get('trees', [])
    
    # Skip if no valid tree structure
    if not trees or len(trees) == 0:
        return None
    if trees[0].get('complete_constituent_tree') is None:
        return None
    
    title = treebank_entry.get('title', 'Unknown')
    chords = treebank_entry.get('chords', [])
    
    # Create the progression string using chord_transition_format
    progression = _format_chord_progression(chords)
    
    # Convert the tree structure to program format
    program = _tree_to_program(trees)
    
    # Return in CorpusRep format
    return {
        "pieceTitle": f"(Valid){title}",
        "program": program,
        "progression": progression
    }


def _format_chord_progression(chords: List[str]) -> str:
    """
    Format a chord progression using chord_transition_format for consecutive pairs.
    
    Args:
        chords: List of chord symbols
    
    Returns:
        Space-separated chord transitions
    """
    if len(chords) < 2:
        return ""
    
    transitions = []
    for i in range(len(chords) - 1):
        transition = chord_transition_format(chords[i], chords[i + 1])
        transitions.append(transition)
    
    return ' '.join(transitions)


def _tree_to_program(trees_list: List[Dict[str, Any]]) -> str:
    """
    Convert the tree structure from treebank to a program representation.
    
    The treebank format has trees as a list of dictionaries with 'complete_constituent_tree'.
    Each tree node has 'label' (chord) and 'children' (list of child nodes).
    
    Since the tree structure represents a nested chord hierarchy rather than explicit
    harmonic operations, this extracts a simplified representation showing the structure.
    
    Args:
        trees_list: A list containing tree dictionaries with 'complete_constituent_tree' key
    
    Returns:
        A program representation string
    """
    if not trees_list or len(trees_list) == 0:
        return "T"
    
    # Get the first tree's complete_constituent_tree
    tree_root = trees_list[0].get('complete_constituent_tree')
    if not tree_root:
        return "T"
    
    # Recursively process the tree
    return _process_tree_node(tree_root)


def _process_tree_node(node: Dict[str, Any]) -> str:
    """
    Recursively process a tree node to build a program structure.
    
    For nodes with children, creates a structure showing:
    - (right-head <chord_transition>) if the node label matches the right child
    - (left-head <chord_transition>) if the node label matches the left child
    
    The chord_transition is computed from the left and right child labels.
    Leaf nodes (no children) are represented as "T".
    """
    if not node or not isinstance(node, dict):
        return "T"
    
    children = node.get('children', [])
    node_label = node.get('label', '')
    
    # Leaf node: no children
    if not children:
        return "T"
    
    # Get labels from children for transition computation
    left_child = children[0]
    right_child = children[-1]  # Rightmost child
    
    left_label = left_child.get('label', '')
    right_label = right_child.get('label', '')
    
    # Determine head direction based on which child matches the node label
    # (removing markers like * from labels for comparison)
    node_base_label = node_label.rstrip('*')
    left_base_label = left_label.rstrip('*')
    right_base_label = right_label.rstrip('*')
    
    if right_base_label == node_base_label:
        head_direction = "rightHeaded"
    else:
        head_direction = "leftHeaded"
    
    # Compute chord transition between left and right children
    transition = chord_transition_format(left_label, right_label)
    
    # Recursively process children
    processed_children = [_process_tree_node(child) for child in children]
    children_str = ' '.join(processed_children)
    
    return f"({head_direction}-{transition} {children_str})"


def format_progression(progression: str) -> str:
    chords = progression.split(' ')
    transitions = []
    for i in range(len(chords) - 1):
        chord1 = chords[i]
        chord2 = chords[i + 1]
        transitions.append(chord_transition_format(chord1, chord2))
    return ' '.join(transitions)

def run(path_treebank,path_output):
    with open(path_treebank, 'r') as f:
        treebank = json.load(f)

    corpus_rep = [convert_treebank_to_corpus_rep(entry) for entry in treebank]
    corpus_rep = [entry for entry in corpus_rep if entry is not None]

    with open(path_output, 'w') as f:
        json.dump(corpus_rep, f, indent=2)
        
