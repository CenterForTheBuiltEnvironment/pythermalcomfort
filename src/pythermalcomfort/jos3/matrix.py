# -*- coding: utf-8 -*-

"""
This code defines a set of functions and constants to model heat exchange and blood flow in different body parts and layers.
"""

import numpy as np


def sub2whole(subarr_list):
    ishape = 0
    jshape = 0
    for subarr in subarr_list:
        ishape += subarr.shape[0]
        jshape += subarr.shape[1]

    wholearr = np.zeros((ishape, jshape))
    i = 0
    j = 0
    for subarr in subarr_list:
        iend = i + subarr.shape[0]
        jend = j + subarr.shape[1]
        wholearr[i:iend, j:jend] = subarr.copy()
        i += subarr.shape[0]
        j += subarr.shape[1]

    return wholearr


BODY_NAMES = [
    "Head",
    "Neck",
    "Chest",
    "Back",
    "Pelvis",
    "Lshoulder",
    "LArm",
    "LHand",
    "RShoulder",
    "RArm",
    "RHand",
    "LThigh",
    "LLeg",
    "LFoot",
    "RThigh",
    "RLeg",
    "RFoot",
]
LAYER_NAMES = ["artery", "vein", "sfvein", "core", "muscle", "fat", "skin"]


def index_order():
    """
    Defines the index's order of the matrix
    Returns
    -------
    indexdict : nested dictionary
        keys are BODY_NAMES and LAYER_NAMES
    """
    # Defines exsisting layers as 1 or None
    indexdict = {}

    for key in ["Head", "Pelvis"]:
        indexdict[key] = {
            "artery": 1,
            "vein": 1,
            "sfvein": None,
            "core": 1,
            "muscle": 1,
            "fat": 1,
            "skin": 1,
        }

    for key in ["Neck", "Chest", "Back"]:
        indexdict[key] = {
            "artery": 1,
            "vein": 1,
            "sfvein": None,
            "core": 1,
            "muscle": None,
            "fat": None,
            "skin": 1,
        }

    for key in BODY_NAMES[5:]:  # limb segments
        indexdict[key] = {
            "artery": 1,
            "vein": 1,
            "sfvein": 1,
            "core": 1,
            "muscle": None,
            "fat": None,
            "skin": 1,
        }

    # Sets ordered indices in the matrix
    indexdict["CB"] = 0
    order_count = 1
    for bn in BODY_NAMES:
        for ln in LAYER_NAMES:
            if not indexdict[bn][ln] is None:
                indexdict[bn][ln] = order_count
                order_count += 1

    return indexdict, order_count


IDICT, NUM_NODES = index_order()


def index_bylayer(layer):
    """
    Get indices of the matrix by the layer name.
    Parameters
    ----------
    layer : str
        Layer name of jos.
        ex) artery, vein, sfvein, core, muscle, fat or skin.
    Returns
    -------
    indices of the matrix : list
    """

    # Gets indices by the layer name
    outindex = []
    for bn in BODY_NAMES:
        for ln in LAYER_NAMES:
            if (layer.lower() == ln) and IDICT[bn][ln]:
                outindex.append(IDICT[bn][ln])
    return outindex


def validindex_bylayer(layer):
    """
    Get indices of the matrix by the layer name.
    Parameters
    ----------
    layer : str
        Layer name of jos.
        ex) artery, vein, sfvein, core, muscle, fat or skin.
    Returns
    -------
    indices of the matrix : list
    """

    # Gets valid indices of the layer name
    outindex = []
    for i, bn in enumerate(BODY_NAMES):
        if IDICT[bn][layer]:
            outindex.append(i)
    return outindex


# Constant parameters of the matrix' indicies
INDEX = {}
VINDEX = {}
for key in LAYER_NAMES:
    INDEX[key] = index_bylayer(key)
    VINDEX[key] = validindex_bylayer(key)


def localarr(bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot):
    """
    Create matrix to calculate heat exchage by blood flow in each segment [W/K]
        1.067 [Wh/(L･K)] * Bloodflow [L/h] = [W/K]
    """
    bf_local = np.zeros((NUM_NODES, NUM_NODES))
    for i, bn in enumerate(BODY_NAMES):
        # Dictionary of indecies in each body segment
        # key = layer name, value = index of matrix
        indexof = IDICT[bn]

        # Common
        bf_local[indexof["core"], indexof["artery"]] = 1.067 * bf_cr[i]  # art to cr
        bf_local[indexof["skin"], indexof["artery"]] = 1.067 * bf_sk[i]  # art to sk
        bf_local[indexof["vein"], indexof["core"]] = 1.067 * bf_cr[i]  # vein to cr
        bf_local[indexof["vein"], indexof["skin"]] = 1.067 * bf_sk[i]  # vein to sk

        # If the segment has a muslce or fat layer
        if not indexof["muscle"] is None:
            bf_local[indexof["muscle"], indexof["artery"]] = (
                1.067 * bf_ms[i]
            )  # art to ms
            bf_local[indexof["vein"], indexof["muscle"]] = (
                1.067 * bf_ms[i]
            )  # vein to ms
        if not indexof["fat"] is None:
            bf_local[indexof["fat"], indexof["artery"]] = (
                1.067 * bf_fat[i]
            )  # art to fat
            bf_local[indexof["vein"], indexof["fat"]] = 1.067 * bf_fat[i]  # vein to fat

        # Only hand
        if i == 7 or i == 10:
            bf_local[indexof["sfvein"], indexof["artery"]] = (
                1.067 * bf_ava_hand
            )  # art to sfvein
        # Only foot
        if i == 13 or i == 16:
            bf_local[indexof["sfvein"], indexof["artery"]] = (
                1.067 * bf_ava_foot
            )  # art to sfvein

    return bf_local


def vessel_bloodflow(bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot):
    """
    Get artery and vein blood flow rate [l/h]
    """
    xbf = bf_cr + bf_ms + bf_fat + bf_sk

    bf_art = np.zeros(17)
    bf_vein = np.zeros(17)

    # Head
    bf_art[0] = xbf[0]
    bf_vein[0] = xbf[0]

    # Neck (+Head)
    bf_art[1] = xbf[1] + xbf[0]
    bf_vein[1] = xbf[1] + xbf[0]

    # Chest
    bf_art[2] = xbf[2]
    bf_vein[2] = xbf[2]

    # Back
    bf_art[3] = xbf[3]
    bf_vein[3] = xbf[3]

    # Pelvis (+Thighs, Legs, Feet, AVA_Feet)
    bf_art[4] = xbf[4] + xbf[11:17].sum() + 2 * bf_ava_foot
    bf_vein[4] = xbf[4] + xbf[11:17].sum() + 2 * bf_ava_foot

    # L.Shoulder (+Arm, Hand, (arteryのみAVA_Hand))
    bf_art[5] = xbf[5:8].sum() + bf_ava_hand
    bf_vein[5] = xbf[5:8].sum()

    # L.Arm (+Hand)
    bf_art[6] = xbf[6:8].sum() + bf_ava_hand
    bf_vein[6] = xbf[6:8].sum()

    # L.Hand
    bf_art[7] = xbf[7] + bf_ava_hand
    bf_vein[7] = xbf[7]

    # R.Shoulder (+Arm, Hand, (arteryのみAVA_Hand))
    bf_art[8] = xbf[8:11].sum() + bf_ava_hand
    bf_vein[8] = xbf[8:11].sum()

    # R.Arm (+Hand)
    bf_art[9] = xbf[9:11].sum() + bf_ava_hand
    bf_vein[9] = xbf[9:11].sum()

    # R.Hand
    bf_art[10] = xbf[10] + bf_ava_hand
    bf_vein[10] = xbf[10]

    # L.Thigh (+Leg, Foot, (arteryのみAVA_Foot))
    bf_art[11] = xbf[11:14].sum() + bf_ava_foot
    bf_vein[11] = xbf[11:14].sum()

    # L.Leg (+Foot)
    bf_art[12] = xbf[12:14].sum() + bf_ava_foot
    bf_vein[12] = xbf[12:14].sum()

    # L.Foot
    bf_art[13] = xbf[13] + bf_ava_foot
    bf_vein[13] = xbf[13]

    # R.Thigh (+Leg, Foot, (arteryのみAVA_Foot))
    bf_art[14] = xbf[14:17].sum() + bf_ava_foot
    bf_vein[14] = xbf[14:17].sum()

    # R.Leg (+Foot)
    bf_art[15] = xbf[15:17].sum() + bf_ava_foot
    bf_vein[15] = xbf[15:17].sum()

    # R.Foot
    bf_art[16] = xbf[16] + bf_ava_foot
    bf_vein[16] = xbf[16]

    return bf_art, bf_vein


def wholebody(bf_art, bf_vein, bf_ava_hand, bf_ava_foot):
    """
    Create matrix to calculate heat exchange by blood flow between segments [W/K]
    """

    def flow(up, down, bloodflow):
        arr = np.zeros((NUM_NODES, NUM_NODES))
        # Coefficient = 1.067 [Wh/L.K]
        arr[down, up] = 1.067 * bloodflow  # Change unit [L/h] to [W/K]
        return arr

    arr83 = np.zeros((NUM_NODES, NUM_NODES))
    # Matrix offsets of segments
    CB = IDICT["CB"]
    Head = IDICT["Head"]["artery"]
    Neck = IDICT["Neck"]["artery"]
    Chest = IDICT["Chest"]["artery"]
    Back = IDICT["Back"]["artery"]
    Pelvis = IDICT["Pelvis"]["artery"]
    Lshoulder = IDICT["Lshoulder"]["artery"]
    LArm = IDICT["LArm"]["artery"]
    LHand = IDICT["LHand"]["artery"]
    RShoulder = IDICT["RShoulder"]["artery"]
    RArm = IDICT["RArm"]["artery"]
    RHand = IDICT["RHand"]["artery"]
    LThigh = IDICT["LThigh"]["artery"]
    LLeg = IDICT["LLeg"]["artery"]
    LFoot = IDICT["LFoot"]["artery"]
    RThigh = IDICT["RThigh"]["artery"]
    RLeg = IDICT["RLeg"]["artery"]
    RFoot = IDICT["RFoot"]["artery"]

    arr83 += flow(CB, Neck, bf_art[1])  # CB to Neck.art
    arr83 += flow(Neck, Head, bf_art[0])  # Neck.art to Head.art
    arr83 += flow(Head + 1, Neck + 1, bf_vein[0])  # Head.vein to Neck.vein
    arr83 += flow(Neck + 1, CB, bf_vein[1])  # Neck.vein to CB

    arr83 += flow(CB, Chest, bf_art[2])  # CB to Chest.art
    arr83 += flow(Chest + 1, CB, bf_vein[2])  # Chest.vein to CB

    arr83 += flow(CB, Back, bf_art[3])  # CB to Back.art
    arr83 += flow(Back + 1, CB, bf_vein[3])  # Back.vein to CB

    arr83 += flow(CB, Pelvis, bf_art[4])  # CB to Pelvis.art
    arr83 += flow(Pelvis + 1, CB, bf_vein[4])  # Pelvis.vein to CB

    arr83 += flow(CB, Lshoulder, bf_art[5])  # CB to Lshoulder.art
    arr83 += flow(Lshoulder, LArm, bf_art[6])  # Lshoulder.art to LArm.art
    arr83 += flow(LArm, LHand, bf_art[7])  # LArm.art to LHand.art
    arr83 += flow(LHand + 1, LArm + 1, bf_vein[7])  # LHand.vein to LArm.vein
    arr83 += flow(LArm + 1, Lshoulder + 1, bf_vein[6])  # LArm.vein to Lshoulder.vein
    arr83 += flow(Lshoulder + 1, CB, bf_vein[5])  # Lshoulder.vein to CB
    arr83 += flow(LHand + 2, LArm + 2, bf_ava_hand)  # LHand.sfvein to LArm.sfvein
    arr83 += flow(
        LArm + 2, Lshoulder + 2, bf_ava_hand
    )  # LArm.sfvein to Lshoulder.sfvein
    arr83 += flow(Lshoulder + 2, CB, bf_ava_hand)  # Lshoulder.sfvein to CB

    arr83 += flow(CB, RShoulder, bf_art[8])  # CB to RShoulder.art
    arr83 += flow(RShoulder, RArm, bf_art[9])  # RShoulder.art to RArm.art
    arr83 += flow(RArm, RHand, bf_art[10])  # RArm.art to RHand.art
    arr83 += flow(RHand + 1, RArm + 1, bf_vein[10])  # RHand.vein to RArm.vein
    arr83 += flow(RArm + 1, RShoulder + 1, bf_vein[9])  # RArm.vein to RShoulder.vein
    arr83 += flow(RShoulder + 1, CB, bf_vein[8])  # RShoulder.vein to CB
    arr83 += flow(RHand + 2, RArm + 2, bf_ava_hand)  # RHand.sfvein to RArm.sfvein
    arr83 += flow(
        RArm + 2, RShoulder + 2, bf_ava_hand
    )  # RArm.sfvein to RShoulder.sfvein
    arr83 += flow(RShoulder + 2, CB, bf_ava_hand)  # RShoulder.sfvein to CB

    arr83 += flow(Pelvis, LThigh, bf_art[11])  # Pelvis to LThigh.art
    arr83 += flow(LThigh, LLeg, bf_art[12])  # LThigh.art to LLeg.art
    arr83 += flow(LLeg, LFoot, bf_art[13])  # LLeg.art to LFoot.art
    arr83 += flow(LFoot + 1, LLeg + 1, bf_vein[13])  # LFoot.vein to LLeg.vein
    arr83 += flow(LLeg + 1, LThigh + 1, bf_vein[12])  # LLeg.vein to LThigh.vein
    arr83 += flow(LThigh + 1, Pelvis + 1, bf_vein[11])  # LThigh.vein to Pelvis
    arr83 += flow(LFoot + 2, LLeg + 2, bf_ava_foot)  # LFoot.sfvein to LLeg.sfvein
    arr83 += flow(LLeg + 2, LThigh + 2, bf_ava_foot)  # LLeg.sfvein to LThigh.sfvein
    arr83 += flow(LThigh + 2, Pelvis + 1, bf_ava_foot)  # LThigh.vein to Pelvis

    arr83 += flow(Pelvis, RThigh, bf_art[14])  # Pelvis to RThigh.art
    arr83 += flow(RThigh, RLeg, bf_art[15])  # RThigh.art to RLeg.art
    arr83 += flow(RLeg, RFoot, bf_art[16])  # RLeg.art to RFoot.art
    arr83 += flow(RFoot + 1, RLeg + 1, bf_vein[16])  # RFoot.vein to RLeg.vein
    arr83 += flow(RLeg + 1, RThigh + 1, bf_vein[15])  # RLeg.vein to RThigh.vein
    arr83 += flow(RThigh + 1, Pelvis + 1, bf_vein[14])  # RThigh.vein to Pelvis
    arr83 += flow(RFoot + 2, RLeg + 2, bf_ava_foot)  # RFoot.sfvein to RLeg.sfvein
    arr83 += flow(RLeg + 2, RThigh + 2, bf_ava_foot)  # RLeg.sfvein to RThigh.sfvein
    arr83 += flow(RThigh + 2, Pelvis + 1, bf_ava_foot)  # RThigh.vein to Pelvis

    return arr83


def remove_bodyname(text):
    """
    Removing the body name from the parameter name.

    Parameters
    ----------
    text : str
        Parameter name

    Returns
    -------
    rtext : str
        Parameter name removed the body name.
    removed : str
        The removed body name

    """

    rtext = text
    removed = None
    for bn in BODY_NAMES:
        if bn in text:
            rtext = rtext.replace(bn, "")
            removed = bn
            break
    return rtext, removed
