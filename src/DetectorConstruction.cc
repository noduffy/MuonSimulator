#ifndef WITH_PB
#define WITH_PB 1
#endif

// PB_COUNT: Pbブロックの個数（1 または 2 のみ許容）
#ifndef PB_COUNT
#define PB_COUNT 2
#endif

#if (PB_COUNT != 1) && (PB_COUNT != 2)
#error "PB_COUNT must be 1 or 2."
#endif

G4VPhysicalVolume* DetectorConstruction::Construct() {
  auto nist = G4NistManager::Instance();
  auto air  = nist->FindOrBuildMaterial("G4_AIR");
  auto sci  = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

  auto worldBox = new G4Box("World", 30*cm, 30*cm, 30*cm);
  auto worldLV  = new G4LogicalVolume(worldBox, air, "WorldLV");
  auto worldPV  = new G4PVPlacement(nullptr, {}, worldLV, "WorldPV", nullptr, false, 0);

  auto plateBox = new G4Box("Plate", 15*cm, 15*cm, 0.5*mm);
  auto plateLV  = new G4LogicalVolume(plateBox, sci, "PlateLV");

  new G4PVPlacement(nullptr, G4ThreeVector(0,0, +8*cm), plateLV, "TopPlate",     worldLV, false, 0);
  new G4PVPlacement(nullptr, G4ThreeVector(0,0, -8*cm), plateLV, "BottomPlate",  worldLV, false, 1);

#if WITH_PB
  // 鉛ブロック（中心 20mm 立方）
  auto pb    = nist->FindOrBuildMaterial("G4_Pb");
  auto pbBox = new G4Box("PbBlock", 10*mm, 10*mm, 10*mm);
  auto pbLV  = new G4LogicalVolume(pbBox, pb, "PbLV");

  // 可視化（LVに対して1回でOK）
  auto visPb = new G4VisAttributes(G4Colour(0.2,0.2,0.2,0.8));
  visPb->SetForceSolid(true);
  pbLV->SetVisAttributes(visPb);

#if PB_COUNT == 1
  // 1個: 中央に配置
  new G4PVPlacement(nullptr, G4ThreeVector(0*mm, 0, 0), pbLV, "PbBlockPV_C", worldLV, false, 0);

#elif PB_COUNT == 2
  // 2個: 左右に配置 (-40mm, +40mm)
  new G4PVPlacement(nullptr, G4ThreeVector(-40*mm, 0, 0), pbLV, "PbBlockPV_L", worldLV, false, 0);
  new G4PVPlacement(nullptr, G4ThreeVector(+40*mm, 0, 0), pbLV, "PbBlockPV_R", worldLV, false, 1);
#endif

#endif // WITH_PB

  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());
  auto visPlate = new G4VisAttributes(G4Colour(0.2,0.8,0.2,0.4));
  plateLV->SetVisAttributes(visPlate);

  return worldPV;
}
