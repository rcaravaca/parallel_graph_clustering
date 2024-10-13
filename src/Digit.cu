#include "Digit.h"

//##########################################################################################
/**
 * @brief Private method to build the ID by concatenating row and col.
 */
void Digit::buildID() {

    ID = ( row * 66 ) + col ;
}

/**
 * @brief Constructor for the Digit class.
 */
Digit::Digit(int r, int c, int e) : row(r), col(c), energy(e) {
    buildID();
}

/**
 * @brief Getter for the row value.
 */
int Digit::getRow() const {
    return row;
}

/**
 * @brief Getter for the column value.
 */
int Digit::getCol() const {
    return col;
}

/**
 * @brief Getter for the energy value.
 */
int Digit::getEnergy() const {
    return energy;
}

/**
 * @brief Getter for the generated ID.
 */
int Digit::getID() const {
    return ID;
}

/**
 * @brief Setter for the row value.
 */
void Digit::setRow(int r) {
    row = r;
    buildID();  // Recalculate ID when row changes
}

/**
 * @brief Setter for the column value.
 */
void Digit::setCol(int c) {
    col = c;
    buildID();  // Recalculate ID when col changes
}

/**
 * @brief Setter for the energy value.
 */
void Digit::setEnergy(int e) {
    energy = e;
}
