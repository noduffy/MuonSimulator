#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"      // 追加
#include "QGSP_BERT.hh"

// あなたのユーザークラス
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

int main(int argc, char** argv) {   // ← 引数あり
  auto* runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

  // 初期化（あなたのクラス名に合わせて）
  runManager->SetUserInitialization(new DetectorConstruction());
  runManager->SetUserInitialization(new QGSP_BERT);
  runManager->SetUserAction(new PrimaryGeneratorAction());
  runManager->SetUserAction(new RunAction());
  runManager->SetUserAction(new SteppingAction());

  runManager->Initialize();

  auto* UImanager = G4UImanager::GetUIpointer();

  if (argc > 1) {
    // 引数で渡されたマクロだけを実行（固定の ../run.mac は呼ばない）
    G4String macro = argv[1];
    UImanager->ApplyCommand("/control/execute " + macro);
  } else {
    // 対話モード
    auto* ui = new G4UIExecutive(argc, argv);
    ui->SessionStart();
    delete ui;
  }

  delete runManager;
  return 0;
}
