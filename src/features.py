import numpy as np
import librosa


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TAXA_AMOSTRAGEM = 16000

NUM_MFCC = 13


# ============================================================
# EXTRAÇÃO DE PITCH
# ============================================================

def extrair_pitch(y, sr):
    """
    Extrai a frequência fundamental (pitch) do áudio.

    Retorna:
        média do pitch
        desvio padrão do pitch
    """

    try:

        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        # Remove valores NaN
        f0 = f0[~np.isnan(f0)]

        if len(f0) == 0:
            return 0.0, 0.0

        media = np.mean(f0)
        desvio = np.std(f0)

        return float(media), float(desvio)

    except Exception:
        return 0.0, 0.0


# ============================================================
# EXTRAÇÃO DE CARACTERÍSTICAS
# ============================================================

def extrair_caracteristicas(caminho_audio):
    """
    Carrega um arquivo WAV e extrai suas características
    acústicas.

    Parâmetros:
        caminho_audio: caminho do arquivo .wav

    Retorna:
        vetor numpy contendo todas as características.
    """

    # --------------------------------------------------------
    # Carregamento do áudio
    # --------------------------------------------------------

    y, sr = librosa.load(
        caminho_audio,
        sr=TAXA_AMOSTRAGEM,
        mono=True
    )

    # --------------------------------------------------------
    # Verificação do áudio
    # --------------------------------------------------------

    if len(y) == 0:
        raise ValueError(
            f"O arquivo de áudio está vazio: {caminho_audio}"
        )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=NUM_MFCC
    )

    mfcc_media = np.mean(
        mfcc,
        axis=1
    )

    mfcc_desvio = np.std(
        mfcc,
        axis=1
    )

    # --------------------------------------------------------
    # Delta MFCC
    # --------------------------------------------------------

    delta_mfcc = librosa.feature.delta(mfcc)

    delta_media = np.mean(
        delta_mfcc,
        axis=1
    )

    delta_desvio = np.std(
        delta_mfcc,
        axis=1
    )

    # --------------------------------------------------------
    # RMS - Energia do sinal
    # --------------------------------------------------------

    rms = librosa.feature.rms(y=y)

    rms_media = np.mean(rms)
    rms_desvio = np.std(rms)

    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_media = np.mean(zcr)
    zcr_desvio = np.std(zcr)

    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    centroid_media = np.mean(centroid)
    centroid_desvio = np.std(centroid)

    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    bandwidth_media = np.mean(bandwidth)
    bandwidth_desvio = np.std(bandwidth)

    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    rolloff_media = np.mean(rolloff)
    rolloff_desvio = np.std(rolloff)

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    pitch_media, pitch_desvio = extrair_pitch(
        y,
        sr
    )

    # --------------------------------------------------------
    # Montagem do vetor final
    # --------------------------------------------------------

    caracteristicas = np.concatenate([

        # MFCC
        mfcc_media,
        mfcc_desvio,

        # Delta MFCC
        delta_media,
        delta_desvio,

        # Energia
        [
            rms_media,
            rms_desvio
        ],

        # Zero Crossing Rate
        [
            zcr_media,
            zcr_desvio
        ],

        # Spectral Centroid
        [
            centroid_media,
            centroid_desvio
        ],

        # Spectral Bandwidth
        [
            bandwidth_media,
            bandwidth_desvio
        ],

        # Spectral Rolloff
        [
            rolloff_media,
            rolloff_desvio
        ],

        # Pitch
        [
            pitch_media,
            pitch_desvio
        ]

    ])

    # --------------------------------------------------------
    # Garantir que não existam NaN ou infinito
    # --------------------------------------------------------

    caracteristicas = np.nan_to_num(
        caracteristicas,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return caracteristicas.astype(np.float32)


# ============================================================
# TESTE DO ARQUIVO
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EXTRATOR DE CARACTERÍSTICAS DE ÁUDIO")
    print("=" * 60)
    print()
    print("Este arquivo contém as funções utilizadas")
    print("pelos modelos de pessoas e emoções.")
    print()
    print("Características extraídas:")
    print()
    print("- MFCC")
    print("- Delta MFCC")
    print("- RMS")
    print("- Zero Crossing Rate")
    print("- Spectral Centroid")
    print("- Spectral Bandwidth")
    print("- Spectral Rolloff")
    print("- Pitch")
    print()
    print("O arquivo está pronto para ser utilizado")
    print("pelos scripts de treinamento.")
    print()

