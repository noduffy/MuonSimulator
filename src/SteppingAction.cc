#include "SteppingAction.hh"
#include "RunAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"

void SteppingAction::UserSteppingAction(const G4Step* step) {
  auto post = step->GetPostStepPoint();
  // ジオメトリ境界（物質に入った瞬間）のみを記録
  if (!post || post->GetStepStatus() != fGeomBoundary) return;

  auto pv = post->GetPhysicalVolume();
  if (!pv) return;
  auto lv = pv->GetLogicalVolume();
  if (!lv) return;
  
  G4String name = lv->GetName();

  // ★修正: "PbLV" (鉛ブロック) は記録しない！
  // 検出器 "PlateLV" へのヒットのみを通す
  if (name != "PlateLV") return;

  const auto* trk = step->GetTrack();
  // 荷電粒子（ミューオン）以外は無視（今回はミューオン単射なら気にしなくてOK）
  if (trk->GetDefinition()->GetPDGCharge() == 0) return;

  auto dir = trk->GetMomentumDirection();
  auto pos = post->GetPosition();

  auto& out = RunAction::Out();
  if (out.good()) {
    const auto* evt = G4EventManager::GetEventManager()->GetConstCurrentEvent();
    int evid = evt ? evt->GetEventID() : -1;
    
    // CSVフォーマット: eventID, trackID, volName, physName, x, y, z, dx, dy, dz
    out << evid << ','
        << trk->GetTrackID() << ','
        << lv->GetName() << ','
        << pv->GetName() << ','
        << pos.x()/mm << ','
        << pos.y()/mm << ','
        << pos.z()/mm << ','
        << dir.x() << ','
        << dir.y() << ','
        << dir.z() << '\n';
  }
}