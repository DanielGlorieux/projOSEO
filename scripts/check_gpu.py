#!/usr/bin/env python3
"""
Script de vérification GPU pour entraînement
Vérifie que PyTorch peut utiliser le GPU
"""
import torch
import sys

def check_gpu():
    """Vérifie disponibilité GPU et configuration"""
    print("="*60)
    print("🔍 VÉRIFICATION CONFIGURATION GPU")
    print("="*60)
    
    # Version PyTorch
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    
    # CUDA disponible?
    cuda_available = torch.cuda.is_available()
    print(f"\n🎮 CUDA Disponible: {'✅ OUI' if cuda_available else '❌ NON'}")
    
    if cuda_available:
        # Détails GPU
        print(f"\n🖥️  GPU Détecté:")
        print(f"  - Nombre de GPUs: {torch.cuda.device_count()}")
        print(f"  - GPU Actif: {torch.cuda.current_device()}")
        print(f"  - Nom GPU: {torch.cuda.get_device_name(0)}")
        
        # Mémoire GPU
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  - Mémoire totale: {total_memory:.2f} GB")
        
        # Version CUDA
        print(f"\n🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Test simple
        print("\n🧪 Test Calcul GPU:")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("  ✅ Calcul GPU fonctionnel!")
            
            # Test performance
            import time
            start = time.time()
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"  ⚡ Performance: {gpu_time:.4f}s pour 100 multiplications matricielles")
            
        except Exception as e:
            print(f"  ❌ Erreur calcul GPU: {e}")
            return False
            
        # Recommandations
        print("\n✅ CONFIGURATION OPTIMALE DÉTECTÉE!")
        print("📊 Configuration entraînement recommandée:")
        print(f"  - Batch size: 256 (GPU) vs 32 (CPU)")
        print(f"  - Workers: 4")
        print(f"  - Mixed precision: Activée")
        print(f"  - Accélération attendue: 5-8x")
        
        return True
        
    else:
        print("\n⚠️  GPU NON DÉTECTÉ - Mode CPU")
        print("\n🔍 Diagnostics:")
        print("  1. Vérifiez nvidia-smi:")
        print("     $ nvidia-smi")
        print("  2. Vérifiez CUDA installé:")
        print("     $ nvcc --version")
        print("  3. Réinstallez PyTorch GPU:")
        print("     $ pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118")
        
        return False

def check_dependencies():
    """Vérifie autres dépendances critiques"""
    print("\n" + "="*60)
    print("📦 VÉRIFICATION DÉPENDANCES")
    print("="*60)
    
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'stable_baselines3',
        'gymnasium'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MANQUANT!")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("\n🚀 ONEA HACKATHON - Vérification Environnement GPU\n")
    
    gpu_ok = check_gpu()
    deps_ok = check_dependencies()
    
    print("\n" + "="*60)
    if gpu_ok and deps_ok:
        print("✅ SYSTÈME PRÊT POUR ENTRAÎNEMENT GPU!")
        print("="*60)
        print("\n🎯 Commandes suivantes:")
        print("  $ python3 scripts/train_models.py --station OUG_ZOG --models all")
        print("\n⚡ Attendez-vous à 5-8x plus rapide qu'en CPU!")
        sys.exit(0)
    elif not gpu_ok and deps_ok:
        print("⚠️  MODE CPU - Entraînement possible mais lent")
        print("="*60)
        print("\n💡 Installez CUDA et PyTorch GPU pour accélération")
        sys.exit(1)
    else:
        print("❌ DÉPENDANCES MANQUANTES")
        print("="*60)
        print("\n🔧 Installez les dépendances:")
        print("  $ pip install -r requirements-gpu-linux.txt")
        sys.exit(1)
