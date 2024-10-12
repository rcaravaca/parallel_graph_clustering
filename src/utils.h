// src/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include "Digit.h"

/**
 * @brief Helper function to trim leading and trailing whitespace from a string.
 * 
 * @param str The input string to be trimmed.
 * @return std::string The trimmed string.
 */
std::string trim(const std::string& str);

/**
 * @brief Function to read a JSON file and return a vector of Event structures.
 * 
 * This function reads the JSON file line by line, extracts event information, 
 * and constructs a vector of Event objects containing the event ID and the associated digits.
 * 
 * @param filename The name of the JSON file to be read.
 * @return std::vector<Event> A vector of Event structures containing event data.
 */
std::vector<Event> readJSON(const std::string& filename);


/**
 * @brief Provides a summary of the energy values of a vector of Digits.
 * 
 * This function calculates and prints the minimum, maximum, average, and
 * standard deviation of the energy values of all Digits. Additionally, it counts
 * how many Digits have an energy value greater than 50.
 * 
 * @param digits The vector of Digits to be analyzed.
 */
void DigitEnergySummary(const std::vector<Digit>& digits);



#endif // UTILS_H
