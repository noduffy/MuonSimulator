#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "Randomize.hh" // G4UniformRand() のために必要

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  // G4ParticleGun の初期化はコンストラクタで行う
  fGun = new G4ParticleGun(1);
  auto muMinus = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  fGun->SetParticleDefinition(muMinus);
  fGun->SetParticleEnergy(3*GeV);
  // 位置と方向は GeneratePrimaries で設定するため、ここでは設定しない
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
  delete fGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  // DetectorConstruction.cc より、検出板の半幅は 10*cm
  G4double half_size = 10.0*cm; 
  G4double z_start = 5.0*cm; // 上部検出板 (3cm) の上から開始

  // X と Y の位置を [-10cm, 10cm] の間でランダムに生成
  G4double rand_x = 2.0 * half_size * G4UniformRand() - half_size;
  G4double rand_y = 2.0 * half_size * G4UniformRand() - half_size;

  // 設定をイベントごとに適用
  fGun->SetParticlePosition({rand_x, rand_y, z_start});
  fGun->SetParticleMomentumDirection({0,0,-1}); // 真っ直ぐ下向きを維持

  fGun->GeneratePrimaryVertex(event);
}