#include <iostream>
#include <string>
#include "gradient.cuh"

// Future functions will be declared here as we progress through the tutorial
// void renderWithVec3();
// void renderSky();
// void renderSphere();
// etc.

void displayMenu() {
    std::cout << "\n=========================================\n";
    std::cout << "  Ray Tracing in One Weekend - CUDA\n";
    std::cout << "=========================================\n\n";
    std::cout << "Select an image to generate:\n\n";
    std::cout << "  1. Simple Gradient (Chapter 2)\n";
    std::cout << "  2. (Coming soon...)\n";
    std::cout << "  3. (Coming soon...)\n";
    std::cout << "\n  0. Exit\n";
    std::cout << "\n=========================================\n";
    std::cout << "Enter your choice: ";
}

int main() {
    int choice = -1;

    while (choice != 0) {
        displayMenu();

        if (!(std::cin >> choice)) {
            // Handle invalid input
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "\nInvalid input! Please enter a number.\n";
            continue;
        }

        std::cout << "\n";

        switch (choice) {
            case 1:
                std::cout << "Generating simple gradient image...\n";
                renderGradient();
                break;

            case 0:
                std::cout << "Exiting program. Goodbye!\n";
                break;

            default:
                std::cout << "Invalid choice! Please select a valid option.\n";
                break;
        }

        if (choice != 0) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore(10000, '\n');
            std::cin.get();
        }
    }

    return 0;
}
