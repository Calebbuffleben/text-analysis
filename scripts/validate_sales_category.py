#!/usr/bin/env python3
"""
Script de validação manual para classificação de categorias de vendas.

Este script permite testar a classificação de categorias de vendas
com textos reais sem precisar executar o serviço completo.

Uso:
    python scripts/validate_sales_category.py "Quanto custa isso?"
"""

import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.models.bert_analyzer import BERTAnalyzer
from src.config import Config


def validate_sales_category(text: str):
    """
    Valida classificação de categoria de vendas para um texto.
    
    Args:
        text: Texto a ser classificado
    """
    print(f"\n{'='*60}")
    print(f"Validando classificação de categoria de vendas")
    print(f"{'='*60}")
    print(f"\nTexto: '{text}'")
    print(f"\nConfiguração:")
    print(f"  - SBERT Model: {Config.SBERT_MODEL_NAME}")
    print(f"  - Device: {Config.MODEL_DEVICE}")
    
    if not Config.SBERT_MODEL_NAME:
        print("\n⚠️  AVISO: SBERT_MODEL_NAME não está configurado!")
        print("   Configure a variável de ambiente SBERT_MODEL_NAME")
        print("   Exemplo: export SBERT_MODEL_NAME='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'")
        return
    
    try:
        # Criar analisador
        print("\n📦 Inicializando BERTAnalyzer...")
        analyzer = BERTAnalyzer(
            model_name=Config.MODEL_NAME,
            device=Config.MODEL_DEVICE,
            cache_dir=Config.MODEL_CACHE_DIR,
            max_length=Config.ANALYSIS_MAX_LENGTH,
            sbert_model_name=Config.SBERT_MODEL_NAME
        )
        
        # Classificar texto
        print("🔍 Classificando texto...")
        categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer.classify_sales_category(
            text,
            min_confidence=0.3
        )
        
        # Exibir resultados
        print(f"\n{'='*60}")
        print("RESULTADOS")
        print(f"{'='*60}")
        
        if categoria:
            print(f"\n✅ Categoria detectada: {categoria}")
            print(f"   Confiança: {confianca:.4f} ({confianca*100:.2f}%)")
            print(f"   Intensidade: {intensidade:.4f} ({intensidade*100:.2f}%)")
            print(f"   Ambiguidade: {ambiguidade:.4f} ({ambiguidade*100:.2f}%)")
            
            # Exibir flags ativas
            active_flags = [flag for flag, value in flags.items() if value]
            if active_flags:
                print(f"\n🚩 Flags semânticas ativas:")
                for flag in active_flags:
                    print(f"   ✓ {flag}")
            else:
                print(f"\n🚩 Flags semânticas: nenhuma ativa")
        else:
            print("\n❌ Nenhuma categoria detectada com confiança suficiente")
            print(f"   (threshold mínimo: 0.3)")
            print(f"   Intensidade: {intensidade:.4f}")
            print(f"   Ambiguidade: {ambiguidade:.4f}")
        
        print(f"\n📊 Scores de todas as categorias:")
        print(f"{'-'*60}")
        
        # Ordenar scores do maior para menor
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (cat, score) in enumerate(sorted_scores, 1):
            marker = "👉" if cat == categoria else "  "
            bar_length = int(score * 40)  # Barra visual
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{marker} {i}. {cat:25s} {score:.4f} │{bar}│")
        
        print(f"{'-'*60}")
        
        # Estatísticas
        if scores:
            best_score = max(scores.values())
            worst_score = min(scores.values())
            avg_score = sum(scores.values()) / len(scores)
            
            print(f"\n📈 Estatísticas:")
            print(f"   Melhor score: {best_score:.4f}")
            print(f"   Pior score: {worst_score:.4f}")
            print(f"   Score médio: {avg_score:.4f}")
            print(f"   Diferença: {best_score - worst_score:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Função principal."""
    if len(sys.argv) < 2:
        print("Uso: python scripts/validate_sales_category.py \"<texto>\"")
        print("\nExemplos:")
        print('  python scripts/validate_sales_category.py "Quanto custa isso?"')
        print('  python scripts/validate_sales_category.py "Como isso vai me ajudar?"')
        print('  python scripts/validate_sales_category.py "Não estou interessado"')
        sys.exit(1)
    
    text = sys.argv[1]
    validate_sales_category(text)


if __name__ == '__main__':
    main()

