"""
Test script to validate debug output system
Tests different stage combinations to ensure debug images are generated correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src import image_processing

def test_single_stage():
    """Test with only preprocessing stage enabled"""
    print("=" * 70)
    print("TEST 1: Testing preprocessing stage only")
    print("=" * 70)

    # Configure debug
    image_processing.DEBUG_MODE = True
    image_processing.DEBUG_STAGES = {
        'preprocessing': True,
        'disc_detection': False,
        'cup_detection': False,
        'mask_generation': False,
        'postprocessing': False,
        'overlays': False,
    }
    image_processing.set_debug_prefix("test1_preprocessing")

    # Load and process image
    image = cv2.imread('/workspace/input_images/image1.jpg')
    if image is None:
        print("ERROR: Could not load image1.jpg")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = image_processing.process_fundus_image(image_rgb)

    # Check if debug files were created
    debug_folder = image_processing.DEBUG_OUTPUT_FOLDER
    expected_files = [
        "test1_preprocessing_stage1a_green_channel.jpg",
        "test1_preprocessing_stage1b_clahe_enhanced.jpg",
        "test1_preprocessing_stage1c_final_preprocessed.jpg"
    ]

    print(f"\nChecking debug output folder: {debug_folder}")
    all_found = True
    for filename in expected_files:
        filepath = os.path.join(debug_folder, filename)
        if os.path.exists(filepath):
            print(f"  ✓ Found: {filename}")
        else:
            print(f"  ✗ Missing: {filename}")
            all_found = False

    if mask is not None:
        print(f"\n✓ Processing succeeded (mask shape: {mask.shape})")
    else:
        print("\n✗ Processing failed (no mask generated)")

    return all_found

def test_all_stages():
    """Test with all stages enabled"""
    print("\n" + "=" * 70)
    print("TEST 2: Testing all stages enabled")
    print("=" * 70)

    # Configure debug - enable all stages
    image_processing.DEBUG_MODE = True
    image_processing.enable_all_debug_stages()
    image_processing.set_debug_prefix("test2_all")

    # Load and process image
    image = cv2.imread('/workspace/input_images/image1.jpg')
    if image is None:
        print("ERROR: Could not load image1.jpg")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = image_processing.process_fundus_image(image_rgb)

    # Check if debug files were created for all stages
    debug_folder = image_processing.DEBUG_OUTPUT_FOLDER

    expected_files = [
        # Preprocessing (3 files)
        "test2_all_stage1a_green_channel.jpg",
        "test2_all_stage1b_clahe_enhanced.jpg",
        "test2_all_stage1c_final_preprocessed.jpg",
        # Disc detection (6 files)
        "test2_all_stage2a_background_mask.jpg",
        "test2_all_stage2b_retinal_mask.jpg",
        "test2_all_stage2c_retinal_only.jpg",
        "test2_all_stage2d_disc_threshold.jpg",
        "test2_all_stage2e_disc_closed.jpg",
        "test2_all_stage2f_disc_candidates.jpg",
        # Cup detection (4 files)
        "test2_all_stage3a_disc_roi.jpg",
        "test2_all_stage3b_cup_threshold.jpg",
        "test2_all_stage3c_cup_closed.jpg",
        "test2_all_stage3d_cup_candidates.jpg",
        # Mask generation (1 file)
        "test2_all_stage4_combined_mask.jpg",
        # Postprocessing (3 files)
        "test2_all_stage5a_disc_cleaned.jpg",
        "test2_all_stage5b_cup_cleaned.jpg",
        "test2_all_stage5c_final_mask.jpg",
        # Overlays (4 files)
        "test2_all_overlay_disc.jpg",
        "test2_all_overlay_cup.jpg",
        "test2_all_overlay_both.jpg",
        "test2_all_overlay_mask.jpg"
    ]

    print(f"\nChecking debug output folder: {debug_folder}")
    print(f"Expected {len(expected_files)} debug images\n")

    found_count = 0
    missing_files = []

    for filename in expected_files:
        filepath = os.path.join(debug_folder, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
            found_count += 1
        else:
            print(f"  ✗ {filename}")
            missing_files.append(filename)

    print(f"\nResult: {found_count}/{len(expected_files)} files found")

    if mask is not None:
        print(f"✓ Processing succeeded (mask shape: {mask.shape})")
    else:
        print("✗ Processing failed (no mask generated)")

    return len(missing_files) == 0

def test_selective_stages():
    """Test with selective stages (disc and cup detection only)"""
    print("\n" + "=" * 70)
    print("TEST 3: Testing selective stages (disc + cup detection)")
    print("=" * 70)

    # Configure debug
    image_processing.DEBUG_MODE = True
    image_processing.DEBUG_STAGES = {
        'preprocessing': False,
        'disc_detection': True,
        'cup_detection': True,
        'mask_generation': False,
        'postprocessing': False,
        'overlays': False,
    }
    image_processing.set_debug_prefix("test3_selective")

    # Load and process image
    image = cv2.imread('/workspace/input_images/image1.jpg')
    if image is None:
        print("ERROR: Could not load image1.jpg")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = image_processing.process_fundus_image(image_rgb)

    # Check expected files
    debug_folder = image_processing.DEBUG_OUTPUT_FOLDER
    expected_files = [
        "test3_selective_stage2a_background_mask.jpg",
        "test3_selective_stage2b_retinal_mask.jpg",
        "test3_selective_stage2c_retinal_only.jpg",
        "test3_selective_stage2d_disc_threshold.jpg",
        "test3_selective_stage2e_disc_closed.jpg",
        "test3_selective_stage2f_disc_candidates.jpg",
        "test3_selective_stage3a_disc_roi.jpg",
        "test3_selective_stage3b_cup_threshold.jpg",
        "test3_selective_stage3c_cup_closed.jpg",
        "test3_selective_stage3d_cup_candidates.jpg",
    ]

    unexpected_files = [
        "test3_selective_stage1a_green_channel.jpg",  # Should not exist
        "test3_selective_overlay_disc.jpg"  # Should not exist
    ]

    print(f"\nChecking expected files:")
    all_found = True
    for filename in expected_files:
        filepath = os.path.join(debug_folder, filename)
        if os.path.exists(filepath):
            print(f"  ✓ Found: {filename}")
        else:
            print(f"  ✗ Missing: {filename}")
            all_found = False

    print(f"\nVerifying disabled stages don't generate output:")
    none_found = True
    for filename in unexpected_files:
        filepath = os.path.join(debug_folder, filename)
        if os.path.exists(filepath):
            print(f"  ✗ Unexpected file found: {filename}")
            none_found = False
        else:
            print(f"  ✓ Correctly not generated: {filename}")

    if mask is not None:
        print(f"\n✓ Processing succeeded (mask shape: {mask.shape})")
    else:
        print("\n✗ Processing failed (no mask generated)")

    return all_found and none_found

def main():
    """Run all tests"""
    print("DEBUG OUTPUT SYSTEM VALIDATION")
    print("=" * 70)
    print()

    # Run tests
    test1_passed = test_single_stage()
    test2_passed = test_all_stages()
    test3_passed = test_selective_stages()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Single stage): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (All stages): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Test 3 (Selective stages): {'✓ PASSED' if test3_passed else '✗ FAILED'}")

    all_passed = test1_passed and test2_passed and test3_passed
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED - Debug system working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Review debug implementation")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
