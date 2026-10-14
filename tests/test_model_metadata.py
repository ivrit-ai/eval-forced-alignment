"""Test that model metadata (description, model_link, github_link, license)
are properly loaded from aligners and injected into result.json."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from efa import registry, scoring


class TestModelMetadata(unittest.TestCase):
    def test_registry_load(self):
        aligners, derived = registry.load()

        # Check wav2vec2-hebrew
        self.assertIn("wav2vec2-hebrew", aligners)
        w2v = aligners["wav2vec2-hebrew"]
        self.assertEqual(w2v.model_link, "https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew")
        self.assertIsNone(w2v.github_link)
        self.assertIsNone(w2v.license)
        self.assertTrue(w2v.description)

        # Check mms
        self.assertIn("mms", aligners)
        mms = aligners["mms"]
        self.assertEqual(mms.model_link, "https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner")
        self.assertEqual(mms.github_link, "https://github.com/facebookresearch/fairseq")
        self.assertEqual(mms.license, "cc-by-nc-4.0")

        # Check whisper-stable-ts
        self.assertIn("whisper-stable-ts", aligners)
        whisper = aligners["whisper-stable-ts"]
        self.assertEqual(whisper.model_link, "https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2")
        self.assertEqual(whisper.github_link, "https://github.com/jianfch/stable-ts")
        self.assertEqual(whisper.license, "apache-2.0")

        # Check mwa-buckeye and mwa-timit
        for name in ("mwa-buckeye", "mwa-timit"):
            self.assertIn(name, aligners)
            a = aligners[name]
            self.assertEqual(a.model_link, "https://github.com/MLSpeech/Multilingual-Word-Aligner")
            self.assertEqual(a.github_link, "https://github.com/MLSpeech/Multilingual-Word-Aligner")
            self.assertEqual(a.license, "cc-by-nc-4.0")

        # Check clap-ipa
        self.assertIn("clap-ipa", aligners)
        clap = aligners["clap-ipa"]
        self.assertEqual(clap.model_link, "https://huggingface.co/anyspeech/ipa-align-base-phone")
        self.assertEqual(clap.github_link, "https://github.com/lingjzhu/clap-ipa")
        self.assertEqual(clap.license, "mit")

        # Check mfa-viter-hebrew-graphemes
        self.assertIn("mfa-viter-hebrew-graphemes", aligners)
        viter = aligners["mfa-viter-hebrew-graphemes"]
        self.assertEqual(viter.model_link, "https://github.com/thewh1teagle/viter/releases/tag/models-v1.0")
        self.assertEqual(viter.github_link, "https://github.com/thewh1teagle/viter")
        self.assertEqual(viter.license, "mit")

        # Check derived mms-corrected
        self.assertIn("mms-corrected", derived)
        mms_corr = derived["mms-corrected"]
        self.assertEqual(mms_corr.model_link, "https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner")
        self.assertEqual(mms_corr.github_link, "https://github.com/facebookresearch/fairseq")
        self.assertEqual(mms_corr.license, "cc-by-nc-4.0")

    def test_result_json_injection(self):
        """Simulate creating a result.json without running real evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir)
            aligners, derived = registry.load()

            # Fake status dictionary as constructed in efa/run.py
            status = {}
            for name in [*aligners, *derived]:
                st = status.setdefault(name, {})
                obj = aligners.get(name) or derived.get(name)
                st["status"] = "ok"
                st["description"] = obj.description
                st["model_link"] = obj.model_link
                st["github_link"] = obj.github_link
                st["license"] = obj.license
                st["aligned"] = 1
                st["missing"] = 0
                st["failed"] = []

            # Fake evaluated result
            entries = [{"id": "c1", "text": "שלום עולם", "metadata": {}, "labels": []}]
            marks = [{"id": "c1", "annotator": "annotator1",
                      "words": [{"word": "שלום", "start": 0.1, "end": 0.5},
                                {"word": "עולם", "start": 0.6, "end": 1.0}]}]
            pair_labels = {
                "wav2vec2-hebrew": {("c1", "annotator1"): [{"word": "שלום", "start": 0.11, "end": 0.51},
                                                            {"word": "עולם", "start": 0.59, "end": 0.99}]}
            }
            res = scoring.evaluate(entries, marks, pair_labels=pair_labels, draws=10)

            # Inject model details into scored aligners as done in efa/run.py
            for source, stats in res.get("aligners", {}).items():
                st_obj = status.get(source, {})
                obj = aligners.get(source) or derived.get(source)
                stats["description"] = st_obj.get("description") or (obj.description if obj else "")
                stats["model_link"] = st_obj.get("model_link") if "model_link" in st_obj else (obj.model_link if obj else None)
                stats["github_link"] = st_obj.get("github_link") if "github_link" in st_obj else (obj.github_link if obj else None)
                stats["license"] = st_obj.get("license") if "license" in st_obj else (obj.license if obj else None)

            full_result = {
                "schema": "efa-result/1",
                "title": "fake test run",
                "aligner_status": status,
                **res,
            }

            out = run / "result.json"
            out.write_text(json.dumps(full_result, ensure_ascii=False, indent=1), encoding="utf-8")

            # Read back and verify
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("aligner_status", loaded)

            # Check that every model in aligner_status has the 4 keys
            for model_name, st in loaded["aligner_status"].items():
                self.assertIn("description", st)
                self.assertIn("model_link", st)
                self.assertIn("github_link", st)
                self.assertIn("license", st)

            # Check scored aligner
            scored_w2v = loaded["aligners"]["wav2vec2-hebrew"]
            self.assertEqual(scored_w2v["model_link"], "https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew")
            self.assertIsNone(scored_w2v["github_link"])
            self.assertIsNone(scored_w2v["license"])
            self.assertTrue(scored_w2v["description"])

    def test_run_main_with_no_align(self):
        """Test efa.run with --no-align on a minimal mock dataset and mock labels."""
        import sys
        import numpy as np
        import soundfile as sf
        from efa import run as efa_run

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            dataset_dir = tmppath / "dataset"
            audio_dir = dataset_dir / "audio"
            audio_dir.mkdir(parents=True)
            wav_path = audio_dir / "test1.wav"
            # create 1 second dummy wav
            dummy_wav = np.zeros(16000, dtype=np.float32)
            sf.write(str(wav_path), dummy_wav, 16000)

            # create metadata.jsonl
            meta_row = {
                "audio_file_name": "audio/test1.wav",
                "id": "test1",
                "annotator": "user1",
                "text": "בדיקה",
                "words": [{"word": "בדיקה", "start": 0.1, "end": 0.5}],
                "metadata": {},
            }
            (dataset_dir / "metadata.jsonl").write_text(json.dumps(meta_row) + "\n", encoding="utf-8")

            run_dir = tmppath / "run"
            labels_dir = run_dir / "labels"
            labels_dir.mkdir(parents=True)

            # provide a fake label for wav2vec2-hebrew
            label_row = {
                "id": "test1#user1",
                "words": [{"word": "בדיקה", "start": 0.12, "end": 0.48}],
            }
            (labels_dir / "wav2vec2-hebrew.jsonl").write_text(json.dumps(label_row) + "\n", encoding="utf-8")

            # invoke efa.run main
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run.py",
                    "--input", str(dataset_dir),
                    "--run", str(run_dir),
                    "--no-align",
                    "--draws", "5",
                ]
                efa_run.main()
            finally:
                sys.argv = old_argv

            result_file = run_dir / "result.json"
            self.assertTrue(result_file.exists())
            data = json.loads(result_file.read_text(encoding="utf-8"))

            self.assertIn("aligner_status", data)
            # check wav2vec2-hebrew status
            w2v_status = data["aligner_status"]["wav2vec2-hebrew"]
            self.assertEqual(w2v_status["status"], "ok")
            self.assertEqual(w2v_status["model_link"], "https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew")
            self.assertIsNone(w2v_status["github_link"])
            self.assertIsNone(w2v_status["license"])
            self.assertTrue(w2v_status["description"])

            # check scored aligners section
            self.assertIn("aligners", data)
            self.assertIn("wav2vec2-hebrew", data["aligners"])
            w2v_scored = data["aligners"]["wav2vec2-hebrew"]
            self.assertEqual(w2v_scored["model_link"], "https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew")
            self.assertIsNone(w2v_scored["github_link"])
            self.assertIsNone(w2v_scored["license"])
            self.assertTrue(w2v_scored["description"])

            # check unavailable aligners still get the metadata
            whisper_status = data["aligner_status"]["whisper-stable-ts"]
            self.assertEqual(whisper_status["model_link"], "https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2")
            self.assertEqual(whisper_status["github_link"], "https://github.com/jianfch/stable-ts")
            self.assertEqual(whisper_status["license"], "apache-2.0")
            self.assertTrue(whisper_status["description"])


if __name__ == "__main__":
    unittest.main()
