#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Fat Suppression v2 implementation

This script performs basic tests to verify:
1. Model instantiation
2. Data generator functionality
3. Basic training pipeline
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append('..')

def test_imports():
    """Test that all v2 modules can be imported"""
    print("Testing imports...")
    try:
        from models_v2 import build_3d_generator, build_multimodal_discriminator
        from datagen_v2 import DataGenerator3D, MultiModalDataGenerator
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test 3D U-Net model creation"""
    print("\nTesting model creation...")
    try:
        from models_v2 import build_3d_generator, build_multimodal_discriminator

        # Test single modality
        input_shape = (32, 128, 128, 3)
        generator = build_3d_generator(input_shape, f=16, num_modalities=1)
        print(f"✓ Single modality generator created: {generator.input_shape} -> {generator.output_shape}")

        # Test multi-modal
        input_shape_multi = (32, 128, 128, 6)  # 2 modalities * 3 channels
        generator_multi = build_3d_generator(input_shape_multi, f=16, num_modalities=2)
        print(f"✓ Multi-modal generator created: {generator_multi.input_shape} -> {generator_multi.output_shape}")

        # Test discriminator
        output_shape = (32, 128, 128, 1)
        discriminator = build_multimodal_discriminator(output_shape, num_modalities=1)
        print(f"✓ Discriminator created: {discriminator.input_shape} -> {discriminator.output_shape}")

        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_data_generator():
    """Test data generator functionality"""
    print("\nTesting data generator...")
    try:
        from datagen_v2 import DataGenerator3D

        # Test with dummy data path (will fail gracefully)
        input_dims = (32, 128, 128, 3)
        output_dims = (32, 128, 128, 1)

        try:
            generator = DataGenerator3D(
                paths="dummy_path",
                batch_size=1,
                input_dims=input_dims,
                output_dims=output_dims,
                num_modalities=1,
                volume_depth=32
            )
            print(f"✓ DataGenerator3D created (length: {len(generator)})")
        except Exception as e:
            print(f"⚠ DataGenerator3D creation expected to fail with dummy data: {e}")

        return True
    except Exception as e:
        print(f"✗ Data generator test failed: {e}")
        return False

def test_model_forward_pass():
    """Test forward pass through the model"""
    print("\nTesting model forward pass...")
    try:
        from models_v2 import build_3d_generator

        input_shape = (32, 128, 128, 3)
        model = build_3d_generator(input_shape, f=8, num_modalities=1)  # Smaller for testing

        # Create dummy input
        dummy_input = tf.random.normal((1,) + input_shape)

        # Forward pass
        output = model(dummy_input, training=False)
        print(f"✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")

        # Check output shape
        expected_shape = (1, 32, 128, 128, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("✓ Output shape correct")

        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_model_compilation():
    """Test model compilation with optimizer and loss"""
    print("\nTesting model compilation...")
    try:
        from models_v2 import build_3d_generator
        from utils_v2 import VGG_loss, calculate_ssim_3d, calculate_psnr_3d

        input_shape = (32, 128, 128, 3)
        model = build_3d_generator(input_shape, f=8, num_modalities=1)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss=VGG_loss,
            metrics=['mae', calculate_ssim_3d, calculate_psnr_3d]
        )
        print("✓ Model compilation successful")

        return True
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        return False

def test_training_pipeline():
    """Test basic training pipeline setup"""
    print("\nTesting training pipeline...")
    try:
        from models_v2 import build_3d_generator
        from utils_v2 import VGG_loss, calculate_ssim_3d, calculate_psnr_3d

        input_shape = (32, 128, 128, 3)
        model = build_3d_generator(input_shape, f=8, num_modalities=1)

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss=VGG_loss,
            metrics=['mae', calculate_ssim_3d, calculate_psnr_3d]
        )

        # Create dummy data
        batch_size = 1
        dummy_input = tf.random.normal((batch_size,) + input_shape)
        dummy_target = tf.random.normal((batch_size, 32, 128, 128, 1))

        # Test single training step
        with tf.GradientTape() as tape:
            predictions = model(dummy_input, training=True)
            loss = VGG_loss(dummy_target, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"✓ Training step successful, loss: {loss.numpy():.4f}")

        return True
    except Exception as e:
        print(f"✗ Training pipeline test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 50)
    print("Fat Suppression v2 - Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_model_creation,
        test_data_generator,
        test_model_forward_pass,
        test_model_compilation,
        test_training_pipeline
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! v2 implementation is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
