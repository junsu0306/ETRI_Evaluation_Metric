#!/usr/bin/env python3
"""
Refactored NetVLAD Knowledge Distillation Main Application

This is a clean, modular version of the original main_for_KD.py file.
The code has been refactored into separate modules for better maintainability:

- config.py: Configuration management
- models.py: Model factories and utilities  
- trainer.py: Training and evaluation logic
- utils.py: Utility functions and helpers

Usage:
    python main_refactored.py --mode train --arch resnet-18 --teacher_arch vgg16
    python main_refactored.py --mode test --resume path/to/checkpoint
    python main_refactored.py --mode cluster --arch resnet-18
"""
from __future__ import print_function
import sys
from os import environ

# Import all the refactored modules
from config import Config, create_argument_parser
from trainer import Trainer
from utils import (
    setup_environment, setup_device, load_dataset, setup_data_loaders,
    restore_config_from_checkpoint, create_directories, print_system_info,
    print_config_summary, check_prerequisites, save_results_summary
)


class NetVLADApplication:
    """Main application class for NetVLAD Knowledge Distillation"""
    
    def __init__(self):
        self.config = None
        self.device = None
        self.trainer = None
        self.dataset = None
        
    def parse_arguments(self):
        """Parse command line arguments and setup configuration"""
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Create configuration
        self.config = Config().from_args(args)
        
        # Restore configuration from checkpoint if resuming
        self.config = restore_config_from_checkpoint(self.config, parser)
        
        return self.config
    
    def setup(self):
        """Setup environment, device, and other components"""
        # Print system information
        print_system_info()
        
        # Setup environment (seeds, etc.)
        setup_environment(self.config)
        
        # Setup device
        self.device = setup_device(self.config)
        print(f"===> Using device: {self.device}")
        
        # Create necessary directories
        create_directories(self.config)
        
        # Load dataset
        self.dataset = load_dataset(self.config)
        
        # Print configuration summary
        print_config_summary(self.config)
        
        # Check prerequisites
        check_prerequisites(self.config)
        
        # Create trainer
        self.trainer = Trainer(self.config, self.device)
    
    def run_train_mode(self):
        """Run training mode"""
        print("===> Starting Training Mode")
        
        # Setup data loaders
        train_set, whole_train_set, whole_training_data_loader, whole_test_set = \
            setup_data_loaders(self.config, self.dataset)
        
        # Setup models
        self.trainer.setup_models(train_set, whole_test_set)
        
        # Start training
        self.trainer.train(
            train_set, 
            whole_train_set, 
            whole_training_data_loader, 
            whole_test_set, 
            self.dataset
        )
        
        print("===> Training completed successfully!")
    
    def run_test_mode(self):
        """Run testing mode"""
        print("===> Starting Test Mode")
        
        # Setup data loaders
        _, _, _, whole_test_set = setup_data_loaders(self.config, self.dataset)
        
        # Setup models
        self.trainer.setup_models(whole_test_set=whole_test_set)
        
        # Load checkpoint
        if not self.trainer.load_checkpoint():
            raise RuntimeError("No checkpoint found for testing. Please provide --resume path")
        
        # Run evaluation
        epoch = 1
        recalls = self.trainer.test(whole_test_set, epoch, write_tboard=False)
        
        # Save results
        results = {
            'recalls': recalls,
            'mode': 'test',
            'split': self.config.data.split
        }
        save_results_summary(self.config, results)
        
        print("===> Testing completed successfully!")
        return recalls
    
    def run_cluster_mode(self):
        """Run clustering mode"""
        print("===> Starting Cluster Mode")
        
        # Setup data loaders
        _, whole_train_set, _, _ = setup_data_loaders(self.config, self.dataset)
        
        # Setup models
        self.trainer.setup_models()
        
        # Generate clusters
        self.trainer.get_clusters(whole_train_set)
        
        print("===> Clustering completed successfully!")
    
    def run(self):
        """Main application entry point"""
        try:
            # Parse arguments and setup
            self.parse_arguments()
            self.setup()
            
            # Run based on mode
            if self.config.training.mode.lower() == 'train':
                self.run_train_mode()
                
            elif self.config.training.mode.lower() == 'test':
                recalls = self.run_test_mode()
                
                # Print final results
                print("\n===> Final Test Results:")
                for n, recall in recalls.items():
                    print(f"Recall@{n}: {recall:.4f}")
                    
            elif self.config.training.mode.lower() == 'cluster':
                self.run_cluster_mode()
                
            else:
                raise ValueError(f"Unknown mode: {self.config.training.mode}")
                
        except KeyboardInterrupt:
            print("\n===> Training interrupted by user")
            if self.trainer:
                self.trainer.close()
            sys.exit(1)
            
        except Exception as e:
            print(f"\n===> Error occurred: {str(e)}")
            if self.trainer:
                self.trainer.close()
            raise
            
        finally:
            # Cleanup
            if self.trainer:
                self.trainer.close()


def main():
    """Main function"""
    app = NetVLADApplication()
    app.run()


if __name__ == "__main__":
    main() 