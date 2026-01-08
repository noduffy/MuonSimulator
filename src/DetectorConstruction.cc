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

  // --- 変更点1: Worldを大きくする ---
  // 検出器を大きくするため、世界そのものも広げないとエラーになります。
  // 15cm -> 30cm (60cm立方) に拡大
  auto worldBox = new G4Box("World", 30*cm, 30*cm, 30*cm);
  auto worldLV  = new G4LogicalVolume(worldBox, air, "WorldLV");
  auto worldPV  = new G4PVPlacement(nullptr, {}, worldLV, "WorldPV", nullptr, false, 0);

  // --- 変更点2: 検出器を巨大化 & 配置調整 ---
  // 10cm -> 15cm (30cm四方) に拡大。これで斜めのミューオンを拾います。
  auto plateBox = new G4Box("Plate", 15*cm, 15*cm, 0.5*mm);
  auto plateLV  = new G4LogicalVolume(plateBox, sci, "PlateLV");
  
  // 位置を ±3cm -> ±8cm に変更
  // (検出器が大きくなったので少し離しても角度は稼げますし、回転実験もしやすくなります)
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, +8*cm), plateLV, "TopPlate",    worldLV, false, 0);
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, -8*cm), plateLV, "BottomPlate", worldLV, false, 1);


  #if WITH_PB
    // 鉛ブロック（中心 20mm 立方）
    // 配置は前回の実験（2つ置き）のままにしてあります
    auto pb   = nist->FindOrBuildMaterial("G4_Pb");
    auto pbBox = new G4Box("PbBlock", 10*mm, 10*mm, 10*mm);
    auto pbLV  = new G4LogicalVolume(pbBox, pb, "PbLV");
    
    // 左右に配置 (-40mm, +40mm)
    new G4PVPlacement(nullptr, G4ThreeVector(-40*mm,0,0), pbLV, "PbBlockPV", worldLV, false, 0);
    new G4PVPlacement(nullptr, G4ThreeVector(+40*mm,0,0), pbLV, "PbBlockPV", worldLV, false, 1);

    auto visPb = new G4VisAttributes(G4Colour(0.2,0.2,0.2,0.8));
    visPb->SetForceSolid(true);
    pbLV->SetVisAttributes(visPb);
  #endif

  // 可視化属性
  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
  auto visPlate = new G4VisAttributes(G4Colour(0.2,0.8,0.2,0.4));
  plateLV->SetVisAttributes(visPlate);

  return worldPV;
}