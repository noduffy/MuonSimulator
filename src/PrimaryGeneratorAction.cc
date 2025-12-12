#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "Randomize.hh"
#include "G4ThreeVector.hh"
#include <cmath>

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  fGun = new G4ParticleGun(1);
  auto muMinus = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  fGun->SetParticleDefinition(muMinus);
  fGun->SetParticleEnergy(3*GeV); // エネルギーは一旦固定でOK
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
  delete fGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  // 検出器より少し広い範囲から降らせるのが一般的
  G4double half_size = 15.0*cm; 
  G4double z_start = 5.0*cm; 

  // 位置のランダム化
  G4double rand_x = 2.0 * half_size * G4UniformRand() - half_size;
  G4double rand_y = 2.0 * half_size * G4UniformRand() - half_size;
  fGun->SetParticlePosition(G4ThreeVector(rand_x, rand_y, z_start));

  // --- 角度の決定 (Cos^2分布) ---
  // 天頂角 theta: 水平面への入射頻度 ~ cos^2(theta) * cos(theta) = cos^3
  // 逆関数法: u = rand, cos(theta) = u^(1/4)
  G4double u = G4UniformRand();
  G4double cosTheta = std::pow(u, 0.25); 
  G4double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
  
  // 方位角 phi: 0 ~ 360度
  G4double phi = 360.0 * deg * G4UniformRand();

  // 下向きなので z成分はマイナス
  G4double px = sinTheta * std::cos(phi);
  G4double py = sinTheta * std::sin(phi);
  G4double pz = -cosTheta;

  fGun->SetParticleMomentumDirection(G4ThreeVector(px, py, pz));
  fGun->GeneratePrimaryVertex(event);
}