#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>

#include "image.h"

Image::Image(int width, int height) :
        mWidth(width),
        mHeight(height),
        mPixels(new glm::vec3[width * height]) {
}

Image::Image(const std::string& filename) {
    int channels;
    // The textures should be in the linear space
    float* data = stbi_loadf(filename.c_str(), &mWidth, &mHeight, &channels, 3);

    if (!data) {
        std::cout << "\t[Fail to load image: " + filename + "]" << std::endl;
        throw;
    }
    mPixels = new glm::vec3[mWidth * mHeight];
    memcpy(mPixels, data, mWidth * mHeight * sizeof(glm::vec3));

    if (data) {
        stbi_image_free(data);
    }
}

Image::~Image() {
    if (mPixels)
        delete[] mPixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < mWidth && y < mHeight);
    mPixels[(y * mWidth) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * mWidth * mHeight];
    for (int y = 0; y < mHeight; y++) {
        for (int x = 0; x < mWidth; x++) { 
            int i = y * mWidth + x;
            glm::vec3 pix = glm::clamp(mPixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), mWidth, mHeight, 3, bytes, mWidth * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveJPG(const std::string& baseFilename) {
    uint8_t* bytes = new uint8_t[3 * mWidth * mHeight];
    for (int y = 0; y < mHeight; y++) {
        for (int x = 0; x < mWidth; x++) {
            int i = y * mWidth + x;
            glm::vec3 pix = glm::clamp(mPixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = pix.x;
            bytes[3 * i + 1] = pix.y;
            bytes[3 * i + 2] = pix.z;
        }
    }
    std::string filename = baseFilename + ".jpg";
    stbi_write_jpg(filename.c_str(), mWidth, mHeight, 3, bytes, 90);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), mWidth, mHeight, 3, (const float *) mPixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
