#include "mathUtil.h"

namespace Math {
    bool epsilonCheck(float a, float b) {
        if (fabs(fabs(a) - fabs(b)) < EpsCmp) {
            return true;
        }
        else {
            return false;
        }
    }

    glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
        glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
        glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * Pi / 180.f, glm::vec3(1.f, 0.f, 0.f));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * Pi / 180.f, glm::vec3(0.f, 1.f, 0.f));
        rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * Pi / 180.f, glm::vec3(0.f, 0.f, 1.f));
        glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
        return translationMat * rotationMat * scaleMat;
    }
}