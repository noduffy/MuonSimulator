#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4VisAttributes.hh"

#ifndef WITH_PB
#define WITH_PB 1
#endif

G4VPhysicalVolume* DetectorConstruction::Construct() {
  auto nist = G4NistManager::Instance();
  auto air  = nist->FindOrBuildMaterial("G4_AIR");
  auto sci  = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

  // World（10 cm 立方）
  auto worldBox = new G4Box("World", 15*cm, 15*cm, 15*cm);
  auto worldLV  = new G4LogicalVolume(worldBox, air, "WorldLV");
  auto worldPV  = new G4PVPlacement(nullptr, {}, worldLV, "WorldPV", nullptr, false, 0);

  // 上下検出板（±3 cm）
  auto plateBox = new G4Box("Plate", 10*cm, 10*cm, 0.5*mm);     // 半長さ指定
  auto plateLV  = new G4LogicalVolume(plateBox, sci, "PlateLV");
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, +3*cm), plateLV, "TopPlate",    worldLV, false, 0);
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, -3*cm), plateLV, "BottomPlate", worldLV, false, 1);


  #if WITH_PB
    // 鉛ブロック（中心 20mm 立方）
    auto pb   = nist->FindOrBuildMaterial("G4_Pb");
    auto pbBox = new G4Box("PbBlock", 10*mm, 10*mm, 10*mm);
    auto pbLV  = new G4LogicalVolume(pbBox, pb, "PbLV");
    new G4PVPlacement(nullptr, G4ThreeVector(0,0,0), pbLV, "PbBlockPV", worldLV, false, 0);

    auto visPb = new G4VisAttributes(G4Colour(0.2,0.2,0.2,0.8));
    visPb->SetForceSolid(true);
    pbLV->SetVisAttributes(visPb);
  #endif
  // 可視化属性（任意）
  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
  auto visPlate = new G4VisAttributes(G4Colour(0.2,0.8,0.2,0.4));
  plateLV->SetVisAttributes(visPlate);

  return worldPV;
}
