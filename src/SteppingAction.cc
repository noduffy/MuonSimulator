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
  if (!post || post->GetStepStatus() != fGeomBoundary) return;
  if (!post) return;

  auto pv = post->GetPhysicalVolume();
  if (!pv) return;
  auto lv = pv->GetLogicalVolume();
  if (!lv) return;
  if (lv->GetName() != "PlateLV") return;

  const auto* trk = step->GetTrack();
  auto dir = trk->GetMomentumDirection();
  auto pos = post->GetPosition(); // 境界位置（入射点）

  auto& out = RunAction::Out();
  if (out.good()) {
    const auto* evt = G4EventManager::GetEventManager()->GetConstCurrentEvent();
    int evid = evt ? evt->GetEventID() : -1;
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
