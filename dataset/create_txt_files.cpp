#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int main() {
    // std::string path =
    // "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/labelled/";

    std::ofstream fout;
    fout.open("train.txt");

    for (size_t count = 0; count < 60; ++count) {
        std::stringstream filename;
        filename << "buoy_0000" << count << ".png";
        fout << filename.str() <<"\n";
    }
    fout.close();
    return 0;
}
