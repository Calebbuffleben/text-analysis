"""
Testes unitários para o módulo de detecção de reformulação/teach-back.

Testa todas as funções do módulo signals.reformulation:
- detect_reformulation_markers
- compute_reformulation_marker_score
- apply_solution_reformulation_signal_flag
"""

import pytest
from src.signals.reformulation import (
    detect_reformulation_markers,
    compute_reformulation_marker_score,
    apply_solution_reformulation_signal_flag
)


class TestDetectReformulationMarkers:
    """Testes para detect_reformulation_markers"""
    
    def test_detect_markers_encontra_marcadores(self):
        """Input com marcadores → retorna lista com marcadores na ordem correta"""
        text = "Deixa eu ver se entendi, então vocês fazem X e Y?"
        markers = detect_reformulation_markers(text)
        
        assert isinstance(markers, list)
        assert len(markers) >= 1
        assert "deixa eu ver se entendi" in markers
        assert "então vocês" in markers
    
    def test_detect_markers_nao_encontra(self):
        """Input sem marcadores → retorna []"""
        text = "Ok, entendi"
        markers = detect_reformulation_markers(text)
        
        assert markers == []
    
    def test_detect_markers_mantem_ordem(self):
        """Verificar que ordem do retorno segue ordem do array markers"""
        text = "só pra confirmar, quer dizer que vocês fazem isso"
        markers = detect_reformulation_markers(text)
        
        # Verificar que a ordem segue a ordem no array original
        # O primeiro marcador encontrado deve vir antes
        all_markers_in_text = [
            "só pra confirmar",
            "quer dizer que"
        ]
        
        # Se ambos estão presentes, verificar ordem relativa
        if len(markers) >= 2:
            idx_confirmar = markers.index("só pra confirmar") if "só pra confirmar" in markers else -1
            idx_quer_dizer = markers.index("quer dizer que") if "quer dizer que" in markers else -1
            
            if idx_confirmar != -1 and idx_quer_dizer != -1:
                assert idx_confirmar < idx_quer_dizer
    
    def test_detect_markers_case_insensitive(self):
        """Verificar que DEIXA EU VER funciona (case insensitive)"""
        text = "DEIXA EU VER SE ENTENDI"
        markers = detect_reformulation_markers(text)
        
        assert "deixa eu ver se entendi" in markers
    
    def test_detect_markers_texto_none(self):
        """Testar comportamento com texto None"""
        markers = detect_reformulation_markers(None)
        assert markers == []
    
    def test_detect_markers_texto_vazio(self):
        """Testar comportamento com texto vazio"""
        markers = detect_reformulation_markers("")
        assert markers == []
    
    def test_detect_markers_multiplos_marcadores(self):
        """Testar detecção de múltiplos marcadores"""
        text = "Deixa eu ver se entendi, então quer dizer que vocês fazem isso"
        markers = detect_reformulation_markers(text)
        
        assert len(markers) >= 3
        assert "deixa eu ver se entendi" in markers
        assert "então vocês" in markers or "então o que você está dizendo é" in markers
        assert "quer dizer que" in markers


class TestComputeReformulationMarkerScore:
    """Testes para compute_reformulation_marker_score"""
    
    def test_compute_score_vazio(self):
        """[] → 0.0"""
        score = compute_reformulation_marker_score([])
        assert score == 0.0
    
    def test_compute_score_um_marcador(self):
        """[marker1] → 0.5"""
        score = compute_reformulation_marker_score(["deixa eu ver se entendi"])
        assert score == 0.5
    
    def test_compute_score_dois_marcadores(self):
        """[marker1, marker2] → 1.0"""
        markers = ["deixa eu ver se entendi", "só pra confirmar"]
        score = compute_reformulation_marker_score(markers)
        assert score == 1.0
    
    def test_compute_score_tres_mais(self):
        """[marker1, marker2, marker3] → 1.0 (cap)"""
        markers = ["deixa eu ver se entendi", "só pra confirmar", "quer dizer que"]
        score = compute_reformulation_marker_score(markers)
        assert score == 1.0
    
    def test_compute_score_quatro_mais(self):
        """4+ marcadores → 1.0 (cap mantido)"""
        markers = ["deixa eu ver se entendi", "só pra confirmar", "quer dizer que", "ou seja"]
        score = compute_reformulation_marker_score(markers)
        assert score == 1.0


class TestApplySolutionReformulationSignalFlag:
    """Testes para apply_solution_reformulation_signal_flag"""
    
    def test_apply_flag_score_zero(self):
        """score 0.0 → dict não é modificado"""
        flags = {}
        apply_solution_reformulation_signal_flag(flags, 0.0)
        
        # Dict não deve ter a key
        assert 'solution_reformulation_signal' not in flags
        assert flags == {}
    
    def test_apply_flag_score_negativo(self):
        """score negativo → dict não é modificado"""
        flags = {}
        apply_solution_reformulation_signal_flag(flags, -0.1)
        
        assert 'solution_reformulation_signal' not in flags
    
    def test_apply_flag_score_positivo(self):
        """score > 0.0 → flags['solution_reformulation_signal'] = True"""
        flags = {}
        apply_solution_reformulation_signal_flag(flags, 0.5)
        
        assert 'solution_reformulation_signal' in flags
        assert flags['solution_reformulation_signal'] is True
    
    def test_apply_flag_score_um(self):
        """score 1.0 → flags['solution_reformulation_signal'] = True"""
        flags = {}
        apply_solution_reformulation_signal_flag(flags, 1.0)
        
        assert flags['solution_reformulation_signal'] is True
    
    def test_apply_flag_preserva_outras_flags(self):
        """Não deve sobrescrever outras flags existentes"""
        flags = {'outra_flag': False}
        apply_solution_reformulation_signal_flag(flags, 0.5)
        
        assert flags['solution_reformulation_signal'] is True
        assert flags['outra_flag'] is False
    
    def test_apply_flag_nao_seta_false(self):
        """Não deve setar False quando score é zero"""
        flags = {'outra_flag': True}
        apply_solution_reformulation_signal_flag(flags, 0.0)
        
        # Não deve criar a key
        assert 'solution_reformulation_signal' not in flags
        # Outra flag deve ser preservada
        assert flags['outra_flag'] is True

