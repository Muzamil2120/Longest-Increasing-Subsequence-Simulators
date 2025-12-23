"""
User Input Handling Module

Handles user input for LIS algorithms with validation.
Provides predefined sample inputs and interactive input methods.

Author: DAA Semester Project
Date: December 2025
"""

from typing import List, Tuple, Optional


class PredefinedInputs:
    """Collection of predefined test cases for LIS algorithms."""
    
    SAMPLES = {
        '1': {
            'name': 'Basic Example',
            'array': [10, 9, 2, 5, 3, 7, 101, 18],
            'description': 'Standard example with mixed numbers',
            'expected_length': 4,
            'expected_lis': [2, 3, 7, 101]
        },
        '2': {
            'name': 'Already Sorted',
            'array': [1, 2, 3, 4, 5],
            'description': 'Array already in increasing order',
            'expected_length': 5,
            'expected_lis': [1, 2, 3, 4, 5]
        },
        '3': {
            'name': 'Reverse Sorted',
            'array': [5, 4, 3, 2, 1],
            'description': 'Array in reverse order (worst case)',
            'expected_length': 1,
            'expected_lis': '[any single element]'
        },
        '4': {
            'name': 'With Duplicates',
            'array': [0, 1, 0, 4, 4, 4, 3, 5, 6],
            'description': 'Array with duplicate elements',
            'expected_length': 5,
            'expected_lis': '[0, 1, 3, 5, 6]'
        },
        '5': {
            'name': 'Small Array',
            'array': [3, 10, 2, 1, 20],
            'description': 'Small array example',
            'expected_length': 3,
            'expected_lis': '[3, 10, 20]'
        },
        '6': {
            'name': 'Negative Numbers',
            'array': [-5, -3, -1, 0, 2, 4],
            'description': 'Array with negative numbers',
            'expected_length': 6,
            'expected_lis': '[-5, -3, -1, 0, 2, 4]'
        },
        '7': {
            'name': 'Single Element',
            'array': [42],
            'description': 'Single element array',
            'expected_length': 1,
            'expected_lis': '[42]'
        },
        '8': {
            'name': 'Two Elements',
            'array': [5, 3],
            'description': 'Two element array (non-increasing)',
            'expected_length': 1,
            'expected_lis': '[any single element]'
        },
        '9': {
            'name': 'Large Increasing',
            'array': [1, 3, 6, 7, 9, 4, 10, 5, 8],
            'description': 'Medium-sized mixed array',
            'expected_length': 5,
            'expected_lis': '[1, 3, 4, 5, 8]'
        },
        '10': {
            'name': 'All Same',
            'array': [5, 5, 5, 5, 5],
            'description': 'All elements are identical',
            'expected_length': 1,
            'expected_lis': '[5]'
        }
    }
    
    @staticmethod
    def display_samples() -> None:
        """Display all available predefined samples."""
        print("\n" + "=" * 90)
        print("PREDEFINED TEST CASES")
        print("=" * 90)
        
        for key, sample in PredefinedInputs.SAMPLES.items():
            print(f"\n[{key}] {sample['name']}")
            print(f"    Description:     {sample['description']}")
            print(f"    Array:           {sample['array']}")
            print(f"    Expected Length: {sample['expected_length']}")
            if isinstance(sample['expected_lis'], list):
                print(f"    Expected LIS:    {sample['expected_lis']}")
            else:
                print(f"    Expected LIS:    {sample['expected_lis']}")
        
        print("\n" + "=" * 90)
    
    @staticmethod
    def get_sample(choice: str) -> Optional[List[int]]:
        """
        Get a predefined sample by choice.
        
        Args:
            choice: Choice number (1-10)
            
        Returns:
            Array if found, None otherwise
        """
        if choice in PredefinedInputs.SAMPLES:
            sample = PredefinedInputs.SAMPLES[choice]
            return sample['array'].copy()
        return None
    
    @staticmethod
    def get_sample_info(choice: str) -> Optional[dict]:
        """
        Get full info about a predefined sample.
        
        Args:
            choice: Choice number (1-10)
            
        Returns:
            Sample dictionary if found, None otherwise
        """
        return PredefinedInputs.SAMPLES.get(choice)


class InputValidator:
    """Validates and parses user input."""
    
    @staticmethod
    def validate_integer(value: str) -> Tuple[bool, Optional[int], str]:
        """
        Validate if string is a valid integer.
        
        Args:
            value: String to validate
            
        Returns:
            Tuple of (is_valid, integer_value, error_message)
        """
        try:
            num = int(value.strip())
            return True, num, ""
        except ValueError:
            return False, None, f"'{value}' is not a valid integer"
    
    @staticmethod
    def validate_array_string(input_str: str) -> Tuple[bool, Optional[List[int]], str]:
        """
        Validate and parse array input string.
        
        Accepts multiple formats:
        - Space-separated: "1 2 3 4 5"
        - Comma-separated: "1,2,3,4,5"
        - Bracket format: "[1, 2, 3, 4, 5]"
        - Mixed: "1, 2, 3 4, 5"
        
        Args:
            input_str: Input string from user
            
        Returns:
            Tuple of (is_valid, array, error_message)
        """
        if not input_str or not input_str.strip():
            return False, None, "Input cannot be empty"
        
        # Remove brackets if present
        input_str = input_str.strip()
        if input_str.startswith('[') and input_str.endswith(']'):
            input_str = input_str[1:-1]
        
        # Replace commas with spaces for uniform parsing
        input_str = input_str.replace(',', ' ')
        
        # Split by whitespace
        parts = input_str.split()
        
        if not parts:
            return False, None, "No numbers found in input"
        
        # Try to convert each part to integer
        result = []
        errors = []
        
        for i, part in enumerate(parts):
            is_valid, num, error = InputValidator.validate_integer(part)
            if is_valid:
                result.append(num)
            else:
                errors.append(f"Position {i+1}: {error}")
        
        if errors:
            error_msg = "Failed to parse some values:\n  " + "\n  ".join(errors)
            return False, None, error_msg
        
        return True, result, ""
    
    @staticmethod
    def validate_array(arr: List[int]) -> Tuple[bool, str]:
        """
        Validate array constraints.
        
        Args:
            arr: Array to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not arr:
            return False, "Array cannot be empty"
        
        if len(arr) > 100000:
            return False, f"Array too large ({len(arr)} > 100000)"
        
        # Check if all elements are valid integers
        for elem in arr:
            if not isinstance(elem, int):
                return False, f"Array contains non-integer: {elem}"
        
        # Check range (within reasonable bounds)
        if any(abs(x) > 10**9 for x in arr):
            return False, "Array values exceed reasonable bounds (-10^9 to 10^9)"
        
        return True, ""


class InputHandler:
    """Main handler for user input with menu system."""
    
    @staticmethod
    def show_menu() -> str:
        """
        Display main menu and get user choice.
        
        Returns:
            User's choice as string
        """
        print("\n" + "=" * 90)
        print("LONGEST INCREASING SUBSEQUENCE - INPUT SELECTION")
        print("=" * 90)
        
        print("\nChoose input method:")
        print("  [1] Use predefined sample")
        print("  [2] Enter custom array")
        print("  [3] Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        return choice
    
    @staticmethod
    def get_predefined_input() -> Optional[List[int]]:
        """
        Get input from predefined samples.
        
        Returns:
            Selected array or None if cancelled
        """
        PredefinedInputs.display_samples()
        
        while True:
            choice = input("Enter sample number (1-10) or 'back' to go back: ").strip()
            
            if choice.lower() == 'back':
                return None
            
            if choice not in PredefinedInputs.SAMPLES:
                print(f"✗ Invalid choice. Please enter a number between 1 and 10.")
                continue
            
            sample = PredefinedInputs.get_sample_info(choice)
            print(f"\n✓ Selected: {sample['name']}")
            print(f"  Array: {sample['array']}")
            
            return PredefinedInputs.get_sample(choice)
    
    @staticmethod
    def get_custom_input() -> Optional[List[int]]:
        """
        Get custom array input from user.
        
        Returns:
            Validated array or None if cancelled
        """
        print("\n" + "=" * 90)
        print("ENTER CUSTOM ARRAY")
        print("=" * 90)
        
        print("\nInput format examples:")
        print("  • Space-separated:  10 9 2 5 3 7 101 18")
        print("  • Comma-separated:  10, 9, 2, 5, 3, 7")
        print("  • Bracket format:   [10, 9, 2, 5, 3]")
        print("  • Mixed:            10, 9 2, 5 3 7")
        print("\nEnter 'back' to return to main menu.")
        
        while True:
            user_input = input("\nEnter array: ").strip()
            
            if user_input.lower() == 'back':
                return None
            
            # Validate array string format
            is_valid, arr, error = InputValidator.validate_array_string(user_input)
            
            if not is_valid:
                print(f"✗ Error: {error}")
                print("Please try again.")
                continue
            
            # Validate array constraints
            is_valid, error = InputValidator.validate_array(arr)
            
            if not is_valid:
                print(f"✗ Error: {error}")
                print("Please try again.")
                continue
            
            # Success
            print(f"\n✓ Valid array: {arr}")
            print(f"  Length: {len(arr)}")
            return arr
    
    @staticmethod
    def interactive_input() -> Optional[List[int]]:
        """
        Interactive input handler with menu system.
        
        Returns:
            Selected array or None if user exits
        """
        while True:
            choice = InputHandler.show_menu()
            
            if choice == '1':
                arr = InputHandler.get_predefined_input()
                if arr is not None:
                    return arr
            
            elif choice == '2':
                arr = InputHandler.get_custom_input()
                if arr is not None:
                    return arr
            
            elif choice == '3':
                print("\nGoodbye! 👋")
                return None
            
            else:
                print(f"✗ Invalid choice. Please enter 1, 2, or 3.")


# Example usage
if __name__ == "__main__":
    print("\n" + "█" * 90)
    print("█" + "  LIS INPUT HANDLER DEMO".center(88) + "█")
    print("█" * 90)
    
    # Test 1: Display samples
    print("\n[TEST 1] Displaying predefined samples...")
    PredefinedInputs.display_samples()
    
    # Test 2: Validate various input formats
    print("\n[TEST 2] Testing input validation...")
    test_inputs = [
        "1 2 3 4 5",
        "10, 9, 2, 5, 3",
        "[10, 9, 2, 5, 3]",
        "10, 9 2, 5 3",
        "invalid",
        "",
    ]
    
    for test in test_inputs:
        is_valid, arr, error = InputValidator.validate_array_string(test)
        if is_valid:
            print(f"  ✓ '{test}' → {arr}")
        else:
            print(f"  ✗ '{test}' → Error: {error}")
    
    # Test 3: Interactive mode
    print("\n[TEST 3] Starting interactive input handler...")
    print("(Follow the prompts to select or enter an array)\n")
    
    arr = InputHandler.interactive_input()
    
    if arr:
        print(f"\n✓ Final selection: {arr}")
    else:
        print("\nNo array selected.")
    
    print("\n" + "█" * 90 + "\n")
