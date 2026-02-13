#include "DetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include <fstream>
#include <string>

// YAMLから数値を読み取る汎用関数
double GetConfigValue(const std::string& key) {
    std::ifstream file("../configs/grid3d.yml");
    std::string line;
    if (!file.is_open()) return 0.0;
    while (std::getline(file, line)) {
        if (line.find(key + ":") != std::string::npos) {
            return std::stod(line.substr(line.find(":") + 1));
        }
    }
    return 0.0;
}

G4VPhysicalVolume* DetectorConstruction::Construct() {
  // YAMLから値をロード
  const int pb_count    = (int)GetConfigValue("pb_count");
  const int pb_hollow   = (int)GetConfigValue("pb_hollow");
  const double outerHalfVal = GetConfigValue("outer_half");
  const double innerHalfVal = GetConfigValue("inner_half");

  auto nist = G4NistManager::Instance();
  auto air  = nist->FindOrBuildMaterial("G4_AIR");
  auto sci  = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
  auto pb   = nist->FindOrBuildMaterial("G4_Pb");

  // ---- World ----
  auto worldBox = new G4Box("World", 30*cm, 30*cm, 30*cm);
  auto worldLV  = new G4LogicalVolume(worldBox, air, "WorldLV");
  auto worldPV  = new G4PVPlacement(nullptr, {}, worldLV, "WorldPV", nullptr, false, 0);

  // ---- Detector plates ----
  auto plateBox = new G4Box("Plate", 15*cm, 15*cm, 0.5*mm);
  auto plateLV  = new G4LogicalVolume(plateBox, sci, "PlateLV");
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, 8*cm), plateLV, "TopPlate", worldLV, false, 0);
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, -8*cm), plateLV, "BottomPlate", worldLV, false, 1);

  // --- Dimensions ---
  const auto outerHalf = outerHalfVal * mm;
  const auto innerHalf = innerHalfVal * mm;
  const auto wallHalfT = (outerHalf - innerHalf) / 2.0;

  // ---- Logical Volumes ----
  auto solidBlockS = new G4Box("SolidBlock_S", outerHalf, outerHalf, outerHalf);
  auto solidBlockLV = new G4LogicalVolume(solidBlockS, pb, "SolidBlock_LV");

  auto wallLRS = new G4Box("WallLR_S", wallHalfT, outerHalf, outerHalf);
  auto wallFBS = new G4Box("WallFB_S", innerHalf, wallHalfT, outerHalf);
  auto wallLR_LV = new G4LogicalVolume(wallLRS, pb, "WallLR_LV");
  auto wallFB_LV = new G4LogicalVolume(wallFBS, pb, "WallFB_LV");

  // Vis Attributes
  auto visPb = new G4VisAttributes(G4Colour(0.2, 0.2, 0.2, 0.85));
  visPb->SetForceSolid(true);
  solidBlockLV->SetVisAttributes(visPb);
  wallLR_LV->SetVisAttributes(visPb);
  wallFB_LV->SetVisAttributes(visPb);

  // ---- Placement Function ----
  auto placePbBlock = [&](const G4ThreeVector& basePos, const G4String& namePrefix, int copyBase) {
    if (pb_hollow == 1) {
      const auto off = innerHalf + wallHalfT;
      new G4PVPlacement(nullptr, basePos + G4ThreeVector(-off, 0, 0), wallLR_LV, namePrefix + "_WallL", worldLV, false, copyBase + 0);
      new G4PVPlacement(nullptr, basePos + G4ThreeVector(+off, 0, 0), wallLR_LV, namePrefix + "_WallR", worldLV, false, copyBase + 1);
      new G4PVPlacement(nullptr, basePos + G4ThreeVector(0, +off, 0), wallFB_LV, namePrefix + "_WallF", worldLV, false, copyBase + 2);
      new G4PVPlacement(nullptr, basePos + G4ThreeVector(0, -off, 0), wallFB_LV, namePrefix + "_WallB", worldLV, false, copyBase + 3);
    } else {
      new G4PVPlacement(nullptr, basePos, solidBlockLV, namePrefix + "_SolidBlock", worldLV, false, copyBase);
    }
  };

  // ---- Placement ----
  if (pb_count == 1) {
    placePbBlock(G4ThreeVector(0,0,0), "Pb_C", 100);
  } else if (pb_count == 2) {
    placePbBlock(G4ThreeVector(-40*mm, 0, 0), "Pb_L", 100);
    placePbBlock(G4ThreeVector( 40*mm, 0, 0), "Pb_R", 200);
  }

  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
  auto visPlate = new G4VisAttributes(G4Colour(0.2, 0.8, 0.2, 0.4));
  plateLV->SetVisAttributes(visPlate);

  return worldPV;
}