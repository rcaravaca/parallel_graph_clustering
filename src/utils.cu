// src/utils.cu
#include "utils.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iostream>

#include <vector>
#include <cmath>
#include <limits>

std::string trim(const std::string& str) {
    const char* whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return ""; // Return an empty string if only whitespace is found
    }
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

std::vector<Event> readJSON(const std::string& filename) {
    
    std::ifstream file(filename);
    std::vector<Event> events;

    if (!file.is_open()) {
        std::cerr << "Could not open the JSON file." << std::endl;
        return events;  // Return an empty vector if the file could not be opened
    }

    std::string line;
    int event_id = -1;
    bool in_digits_section = false;

    // Variables to store digit values
    int row, col, energy;

    Event current_event;

    // Read the file line by line
    while (getline(file, line)) {

        std::string line_trimmed = trim(line);
        
        // Look for the event ID
        if (line_trimmed.find("\"Event\":") != std::string::npos) { //not find a match
            // If we already have an event with digits, add it to the vector
            if (current_event.event_id != -1) {
                events.push_back(current_event);
                current_event.digits.clear();  // Clear the digits for the next event
            }

            // Create a new event
            std::stringstream ss(line_trimmed);
            std::string variable;
            ss >> variable >> event_id;
            current_event.event_id = event_id;  // Assign the event ID
            in_digits_section = false;
        }

        // Look for the start of the digits section
        if (line_trimmed.find("\"digits\":") != std::string::npos) {
            in_digits_section = true;
        }

        // Process the digits
        if (in_digits_section) {
            
            if (line_trimmed.find("\"row\":") != std::string::npos) {
                std::stringstream ss(line_trimmed);
                std::string variable;
                ss >> variable >> row;
            }

            if (line_trimmed.find("\"col\":") != std::string::npos) {
                std::stringstream ss(line_trimmed);
                std::string variable;
                ss >> variable >> col;
            }

            if (line_trimmed.find("\"energy\":") != std::string::npos) {
                std::stringstream ss(line_trimmed);
                std::string variable;
                ss >> variable >> energy;

                // Add the digit to the current event
                current_event.digits.push_back(Digit{row, col, energy});
                // in_digits_section = false;
                
               
            }




        }// end if Process the digits
    }// end of while

    // Add the last event read to the vector of events
    if (current_event.event_id != -1) {
        events.push_back(current_event);
    }

    // erase the first element
    if (!events.empty()) {
        events.erase(events.begin());
    }

    file.close();
    return events;  // Return the vector of events
}

void DigitEnergySummary(const std::vector<Digit>& digits) {
    // Verificar si el vector está vacío
    if (digits.empty()) {
        std::cout << "No digits available." << std::endl;
        return;
    }

    // Variables para los cálculos
    int numDigits = 0;
    int minEnergy = std::numeric_limits<int>::max();
    int maxEnergy = std::numeric_limits<int>::min();
    double sumEnergy = 0;
    double sumEnergySquared = 0;  // Para calcular la desviación estándar
    int countGreaterThan50 = 0;   // Contador de Digits con energía > 50

    // Recorrer todos los Digits y calcular los valores de energía
    for (const auto& digit : digits) {
        int energy = digit.getEnergy();
        numDigits++;

        // Contar Digits con energía > 50
        if (energy > 50) {
            countGreaterThan50++;
        }

        // Actualizar el valor mínimo y máximo de energía
        if (energy < minEnergy) {
            minEnergy = energy;
        }
        if (energy > maxEnergy) {
            maxEnergy = energy;
        }

        // Acumular la suma de las energías y la suma de las energías al cuadrado
        sumEnergy += energy;
        sumEnergySquared += energy * energy;
    }

    // Cálculos finales
    double meanEnergy = sumEnergy / numDigits;
    double variance = (sumEnergySquared / numDigits) - (meanEnergy * meanEnergy);
    double stdDeviation = sqrt(variance);

    // Imprimir el resumen de las energías
    std::cout << "\n##############################" << std::endl;
    std::cout << "Digit Energy Summary:" << std::endl;
    std::cout << "Number of Digits: " << numDigits << std::endl;
    std::cout << "Energy values: " << std::endl;
    std::cout << "  Minimum: " << minEnergy << std::endl;
    std::cout << "  Maximum: " << maxEnergy << std::endl;
    std::cout << "  Average: " << meanEnergy << std::endl;
    std::cout << "  Standard deviation: " << stdDeviation << std::endl;
    std::cout << "Digits with energy > 50: " << countGreaterThan50 << std::endl;
    std::cout << "##############################" << std::endl<< std::endl;

}
