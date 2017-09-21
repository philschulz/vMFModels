'''
Created on Jul 16, 2014

@author: Philip Schulz

I use the original Brown et al. channel terminology here when talking about source and target.
'''

import argparse, sys
from typing import List, Tuple


def __sort_alignment(alignment: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Sort the alignment links by source position.

    :param alignment: An alignment for a sentence pair.
    :return: The input alignment sorted by source positions.
    """
    return sorted(list(alignment), key=lambda entry: (entry[0], entry[1]))


def moses_wrapper(heuristic):
    """
    Produce a function that reads alignment files and outputs their symmetrised version in the Moses format.

    :param heuristic: An alignment heuristic.
    """

    def readFiles(src2tgtFile, tgt2srcFile, extension, file_name):
        """
        Read alignment files and write their symmetrisation to disk.
        :param src2tgtFile: The source to target alignment file.
        :param tgt2scrFile: The target to source alignment file.
        :param extension: The extension of the output file.
        :param file_name: The prefix of the output file.
        """
        extension = "-".join(extension.split("_"))
        with open(src2tgtFile) as src2tgtStream, open(tgt2srcFile) as tgt2srcStream, open(file_name + "." + extension,
                                                                                          "w") as out:
            i = 0
            for src2tgt_links in src2tgtStream:
                i = i + 1
                sys.stderr.write("reading line " + str(i) + "\r")
                src2tgtAlignment = set(
                    [(int(link[0]), int(link[1])) for link in map(lambda x: x.split('-'), src2tgt_links.split())])
                tgt2srcAlignment = set(
                    [(int(link[1]), int(link[0])) for link in map(lambda x: x.split('-'), next(tgt2srcStream).split())])
                merged_alignment = heuristic(src2tgtAlignment, tgt2srcAlignment)
                for link in merged_alignment:
                    out.write(str(link[0]) + "-" + str(link[1]) + " ")
                # end line
                out.write("\n")

    return readFiles


def naacl_wrapper(heuristic: str) -> None:
    """
    Produce a function that reads alignment files and outputs their symmetrised version in the NAACL format.

    :param heuristic: An alignment heuristic.
    """

    def read_files(src2tgtFile: str, tgt2scrFile: str, extension: str, file_name: str) -> None:
        """
        Read alignment files and write their symmetrisation to disk.
        :param src2tgtFile: The source to target alignment file.
        :param tgt2scrFile: The target to source alignment file.
        :param extension: The extension of the output file.
        :param file_name: The prefix of the output file.
        """
        extension = "-".join(extension.split("_"))
        with open(src2tgtFile) as src2tgtStream, open(tgt2scrFile) as tgt2scrStream, open(file_name + "." + extension,
                                                                                          "w") as out:
            sentence_num = 1
            src2tgt_alignment = set()
            tgt2scr_alignment = set()

            for line in src2tgtStream:
                sys.stderr.write("processing sentence " + str(sentence_num) + "\r")
                sys.stderr.flush()
                src2tgtFields = [int(x) for x in line.split(' ')]
                if src2tgtFields[0] == sentence_num:
                    src2tgt_alignment.add(tuple(src2tgtFields[1:]))
                else:
                    for tgt2src_links in tgt2scrStream:
                        tgt2srcFields = [int(x) for x in tgt2src_links.split(' ')]
                        if tgt2srcFields[0] != sentence_num:
                            tgt2scr_alignment.add(tuple(tgt2srcFields[1:]))
                        else:
                            # write current sentence alignment
                            merged_alignment = heuristic(src2tgt_alignment, tgt2scr_alignment)
                            for link in merged_alignment:
                                out.write(str(sentence_num) + " " + str(link[0]) + " " + str(link[1]))
                                out.write("\n")

                            src2tgt_alignment = set()
                            src2tgt_alignment.add(tuple(src2tgtFields[1:]))
                            tgt2scr_alignment = set()
                            tgt2scr_alignment.add(tuple(tgt2srcFields[1:]))
                            sentence_num += 1

    return read_files


@naacl_wrapper
def naacl_union(src2tgt_links, tgt2src_links):
    return __sort_alignment(__union(src2tgt_links, tgt2src_links)[0])


@naacl_wrapper
def naacl_intersection(src2tgt_links, tgt2src_links):
    return __sort_alignment(__intersection(src2tgt_links, tgt2src_links))


@naacl_wrapper
def naacl_grow(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links))


@naacl_wrapper
def naacl_grow_diag(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links, True))


@naacl_wrapper
def naacl_grow_diag_final(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links, True, True))


@naacl_wrapper
def naacl_grow_diag_final_and(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links, True, True, True))


@moses_wrapper
def moses_union(src2tgt_links, tgt2src_links):
    return __sort_alignment(__union(src2tgt_links, tgt2src_links)[0])


def __union(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Take the union of two alignments.

    :param src2tgt_links: The source to target alignment.
    :param tgt2src_links: The target to source alignment.
    :return: The union of the alignments.
    """
    union_alignment = src2tgt_links.union(tgt2src_links)

    return union_alignment, src2tgt_links, tgt2src_links


@moses_wrapper
def moses_intersection(src2tgt_links, tgt2src_links):
    return __sort_alignment(__intersection(src2tgt_links, tgt2src_links))


def __intersection(src2tgt_links, tgt2src_links):
    intersection_alignment = src2tgt_links & tgt2src_links

    return intersection_alignment


def __srctotgt(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Return the source to target alignment.

    :param src2tgt_links: The source to target alignment.
    :param tgt2src_links: The target to source alignment.
    :return: The source to target alignment.
    """
    return src2tgt_links


def __tgttosrc(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Return the target to source alignment.

    :param src2tgt_links: The source to target alignment.
    :param tgt2src_links: The target to source alignment.
    :return: The source to target alignment.
    """
    return tgt2src_links


@moses_wrapper
def moses_srctotgt(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return __sort_alignment(__srctotgt(src2tgt_links, tgt2src_links))


@moses_wrapper
def moses_tgttosrc(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return __sort_alignment(__tgttosrc(src2tgt_links, tgt2src_links))


@naacl_wrapper
def naacl_srctotgt(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return __sort_alignment(__srctotgt(src2tgt_links, tgt2src_links))


@naacl_wrapper
def naacl_tgttosrc(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return __sort_alignment(__tgttosrc(src2tgt_links, tgt2src_links))


@moses_wrapper
def moses_grow_diag(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links, True))


@moses_wrapper
def moses_grow(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links))


@moses_wrapper
def moses_grow_diag_final(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links, True, True))


@moses_wrapper
def moses_grow_diag_final_and(src2tgt_links, tgt2src_links):
    return __sort_alignment(__grow(src2tgt_links, tgt2src_links, True, True, True))


def __grow(src2tgt_links: List[Tuple[int, int]], tgt2src_links: List[Tuple[int, int]], diag: bool = False,
           final: bool = False, final_and: bool = False) -> List[Tuple[int, int]]:
    """
    Use the grow heuristic to symmetrise alignments.
    
    :param src2tgt_links: The source to target alignment.
    :param tgt2src_links: The target to source alignment.
    :param diag: Use grow-diag.
    :param final: Use grow-diag-final.
    :param final_and: Use grow-diag-final-and.
    :return: The symmetrised alignment.
    """
    intersected_alignment = __intersection(src2tgt_links, tgt2src_links)
    union_alignment, src2tgt_links, tgt2src_links = __union(src2tgt_links, tgt2src_links)

    src_candidates = list(set(link[0] for link in union_alignment))
    src_candidates.sort()
    tgt_candidates = list(set(link[1] for link in union_alignment))
    tgt_candidates.sort()

    src_aligned = set(link[0] for link in intersected_alignment)
    tgt_aligned = set(link[1] for link in intersected_alignment)

    grown_alignment = set(intersected_alignment)

    while True:
        added = False
        # need to iterate first over target and then over source language
        # => this is wrongly described in Moses manual
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if (src_pos, tgt_pos) in grown_alignment:

                    for step in [-1, 1]:
                        new_src_pos = src_pos + step
                        if new_src_pos not in src_aligned:
                            new_link = (new_src_pos, tgt_pos)
                            if new_link in union_alignment:
                                grown_alignment.add(new_link)
                                src_aligned.add(new_src_pos)
                                added = True

                    for step in [-1, 1]:
                        new_tgt_pos = tgt_pos + step
                        if new_tgt_pos not in tgt_aligned:
                            new_link = (src_pos, new_tgt_pos)
                            if new_link in union_alignment:
                                grown_alignment.add(new_link)
                                tgt_aligned.add(new_tgt_pos)
                                added = True

                    # extend alignment diagonally -> my version
                    if diag:
                        for src_step in [-1, 1]:
                            for tgt_step in [-1, 1]:
                                new_src_pos = src_pos + src_step
                                new_tgt_pos = tgt_pos + tgt_step
                                if new_src_pos not in src_aligned or new_tgt_pos not in tgt_aligned:
                                    new_link = (new_src_pos, new_tgt_pos)
                                    if new_link in union_alignment:
                                        grown_alignment.add(new_link)
                                        src_aligned.add(new_src_pos)
                                        tgt_aligned.add(new_tgt_pos)
                                        added = True
        if not added:
            break

    if final and not final_and:
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned or tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in src2tgt_links:
                        grown_alignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)

        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned or tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in tgt2src_links:
                        grown_alignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)

    if final_and:
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned and tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in src2tgt_links:
                        grown_alignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)

        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned and tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in tgt2src_links:
                        grown_alignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)

    return grown_alignment


def main():

    heuristics = ["srctotgt", "tgttosrc", "union", "intersection", "grow", "grow-diag", "grow-diag-final", "grow-diag-final-and"]
    formats = ["moses", "naacl"]

    # create commandLinksParser
    commandLineParser = argparse.ArgumentParser("Merges the alignment files produced by aligning "
                                                "a bilingual corpus in both directions. Different heuristics for the merger must be specified. "
                                                "Note that the different mergers are mutually exclusive, meaning that only one of them can and "
                                                "must be specified. The files are merged according to the order in the first file. Usually, this "
                                                "file should have entries of the form 'target-source' (in IBM terminolgy -- in Moses terminology this "
                                                "would be 'source-target'). This is the format that Moses needs for "
                                                "phrase extraction. Hence the second file should have entries of the form 'source-target' (or 'target-source "
                                                "in Moses terms).")

    commandLineParser.add_argument("alignmentFiles",
                                   nargs=2,
                                   help="The two alignment files that have been generated from aligning the corpus "
                                        "in different directions.")

    commandLineParser.add_argument("--filename",
                                   default="aligned",
                                   help="Set the file_name of the output file. An extension corresponding to the "
                                        "chosen symmetrization heuristic will be added.")

    commandLineParser.add_argument("--format",
                                   default="moses",
                                   choices=formats,
                                   nargs=1,
                                   help="Choose the input and output format of the alignment files. "
                                        "Default: %(default)s. Choices: %(choices)s.")

    # add group of alignment heuristics that are mutually exclusive but one of them is required
    heuristics = commandLineParser.add_mutually_exclusive_group(required=True)

    # create intersection argument
    heuristics.add_argument("--intersect",
                            action="store_true",
                            help="Triggers the intersection heuristic for merging. Only alignment points that are "
                                 "present in BOTH alignments are retained. This lead to high-precision, low-recall "
                                 "alignments. Due to the rather unrestrictive nature of the resulting alignment, "
                                 "subsequent steps in the training pipeline will extract a very large amount of "
                                 "phrases.")

    # create union argument
    heuristics.add_argument("--union",
                            action="store_true",
                            help="Triggers the union heuristic for merging. All points that occur in EITHER alignment "
                                 "are retained. This leads to high-recall alignments.")

    # create grow argument
    heuristics.add_argument("--grow",
                            action="store_true",
                            help="Triggers the grow heuristic for merging. Expands the intersected alignment in "
                                 "horizontal and vertical direction until no new points can be added.")

    # create grow-diag argument
    heuristics.add_argument("--grow-diag",
                            action="store_true",
                            help="Triggers the grow-diag heuristic for merging. This merger first executes the "
                                 "intersection merger and then adds any alignment points that are a) neighbouring a "
                                 "retained alignment point and b) present in EITHER of the original alignments.")

    # create grow-diag-final argument
    heuristics.add_argument("--grow-diag-final",
                            action="store_true",
                            help="Triggers the grow-diag-final heuristic for merging. This merger executes the "
                                 "grow-diag merger and then adds alignment points for unaligned word on EITHER side, "
                                 "given that they are aligned in EITHER alignment. This may effectively introduce "
                                 "'isolated' alignment points that are not in the vicinity of any other alignment "
                                 "points.")

    # create grow-diag-final-and argument
    heuristics.add_argument("--grow-diag-final-and",
                            action="store_true",
                            help="Triggers the grow-diag-final-and heuristc for merging. This merger is just like the "
                                 "grow-diag-final merger with the stronger requirement that yet unaligned words may "
                                 "only be linked to other yet unaligned words.")

    args = vars(commandLineParser.parse_args())
    src2tgt_file = args["alignmentFiles"][0]
    tgt2src_file = args["alignmentFiles"][1]
    file_name = args["filename"]
    file_format = args["format"][0].lower()

    if args["srctotgt"]:
        if file_format == "moses":
            moses_srctotgt(src2tgt_file, tgt2src_file, "srctotgt", file_name)
        elif file_format == "naacl":
            naacl_srctotgt(src2tgt_file, tgt2src_file, "srctotgt", file_name)
    elif args["tgttosrc"]:
        if file_format == "moses":
            moses_tgttosrc(src2tgt_file, tgt2src_file, "tgttosrc", file_name)
        elif file_format == "naacl":
            naacl_tgttosrc(src2tgt_file, tgt2src_file, "tgtotsrc", file_name)
    elif args["union"]:
        if file_format == "moses":
            moses_union(src2tgt_file, tgt2src_file, "union", file_name)
        elif file_format == "naacl":
            naacl_union(src2tgt_file, tgt2src_file, "union", file_name)
    elif args["intersect"]:
        if file_format == "moses":
            moses_intersection(src2tgt_file, tgt2src_file, "intersect", file_name)
        elif file_format == "naacl":
            naacl_intersection(src2tgt_file, tgt2src_file, "intersect", file_name)
    elif args["grow"]:
        if file_format == "moses":
            moses_grow(src2tgt_file, tgt2src_file, "grow", file_name)
        elif file_format == "naacl":
            naacl_grow(src2tgt_file, tgt2src_file, "grow", file_name)
    elif args["grow_diag"]:
        if file_format == "moses":
            naacl_grow_diag(src2tgt_file, tgt2src_file, "grow-diag", file_name)
        elif file_format == "naacl":
            naacl_grow_diag(src2tgt_file, tgt2src_file, "grow-diag", file_name)
    elif args["grow_diag_final"]:
        if file_format == "moses":
            moses_grow_diag_final(src2tgt_file, tgt2src_file, "grow-diag-final", file_name)
        elif file_format == "naacl":
            naacl_grow_diag_final(src2tgt_file, tgt2src_file, "grow-diag-final", file_name)
    elif args["grow_diag_final_and"]:
        if file_format == "moses":
            moses_grow_diag_final_and(src2tgt_file, tgt2src_file, "grow-diag-final-and", file_name)
        elif file_format == "naacl":
            naacl_grow_diag_final_and(src2tgt_file, tgt2src_file, "grow_diag-final-and", file_name)


if __name__ == '__main__':
    main()
