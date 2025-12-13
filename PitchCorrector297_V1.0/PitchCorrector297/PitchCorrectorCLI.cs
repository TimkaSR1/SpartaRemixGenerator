using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using NAudio.Wave;

namespace PitchCorrector297
{
    /// <summary>
    /// CLI helper for headless pitch correction to D note.
    /// </summary>
    public static class PitchCorrectorCLI
    {
        // D4 = 293.66 Hz (standard Sparta Remix pitch)
        private const float DEFAULT_FREQ = 293.66f;

        [DllImport("kernel32.dll")]
        private static extern bool AttachConsole(int dwProcessId);
        private const int ATTACH_PARENT_PROCESS = -1;

        public static int Run(string[] args)
        {
            // Attach to parent console for output
            AttachConsole(ATTACH_PARENT_PROCESS);
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: PitchCorrector297.exe <input.wav> <output.wav> [frequency]");
                Console.WriteLine("  frequency: target pitch in Hz (default 293.66 = D4)");
                return 1;
            }

            string inputPath = args[0];
            string outputPath = args[1];
            float freq = DEFAULT_FREQ;

            if (args.Length >= 3)
            {
                if (!float.TryParse(args[2], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out freq))
                {
                    freq = DEFAULT_FREQ;
                }
            }

            if (!File.Exists(inputPath))
            {
                Console.WriteLine($"Error: Input file not found: {inputPath}");
                return 1;
            }

            try
            {
                ProcessAudio(inputPath, outputPath, freq);
                Console.WriteLine($"OK: {outputPath}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                return 1;
            }
        }

        private static void ProcessAudio(string inputPath, string outputPath, float freq)
        {
            using (var afr = new AudioFileReader(inputPath))
            {
                int sampleRate = afr.WaveFormat.SampleRate;
                int channels = afr.WaveFormat.Channels;

                // Read up to 2.5 seconds of audio
                int maxSamples = (int)(sampleRate * 2.5f);
                float[] samples = new float[maxSamples * channels];
                int read = afr.Read(samples, 0, samples.Length);

                // Convert to mono if stereo
                float[] mono;
                if (channels == 2)
                {
                    mono = new float[read / 2];
                    for (int i = 0; i < mono.Length; i++)
                    {
                        mono[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2f;
                    }
                }
                else
                {
                    mono = new float[read];
                    Array.Copy(samples, mono, read);
                }

                // Normalize input
                Normalize(ref mono);

                // Run pitch detection (autocorrelation)
                int[] pitch = new int[280];
                for (int i = 0; i < 280; i++)
                {
                    pitch[i] = Autocorrelation(ref mono, 512 * i, 512, 2048);
                }

                // Fix octave errors
                float[] pitch2 = new float[280];
                for (int i = 0; i < 280; i++)
                {
                    List<int> pp = new List<int>();
                    for (int j = -3; j <= 3; j++)
                    {
                        int idx = j + i;
                        if (idx >= 0 && idx < 280 && pitch[idx] > -1)
                            pp.Add(pitch[idx]);
                    }
                    if (pp.Count == 0) { pitch2[i] = -1; continue; }
                    pp.Sort();
                    int ppp = pp[pp.Count / 2];

                    int newp = -1;
                    int minDiff = int.MaxValue;
                    for (int j = 1; j < 5; j++)
                    {
                        for (int k = 1; k < 5; k++)
                        {
                            int candidate = pitch[i] * j / k;
                            if (Math.Abs(ppp - candidate) < minDiff)
                            {
                                newp = candidate;
                                minDiff = Math.Abs(ppp - candidate);
                            }
                        }
                    }
                    pitch2[i] = newp;
                    if (newp == 0) pitch2[i] = -1;
                }

                // Find first valid pitch index
                int firstPitchIndex = -1;
                for (int i = 0; i < pitch2.Length; i++)
                {
                    if (pitch2[i] > -1)
                    {
                        firstPitchIndex = i;
                        break;
                    }
                }

                if (firstPitchIndex == -1)
                {
                    // No pitch detected, just copy input
                    SaveWav(mono, sampleRate, outputPath);
                    return;
                }

                // Find pitch markers for PSOLA
                List<int> ptchpt = new List<int>();
                float max = 0;
                int maxi = -1;
                int jj = 0;

                while (maxi == -1 && jj < 100)
                {
                    int start = firstPitchIndex * 512 + jj * (int)pitch2[firstPitchIndex];
                    int end = (int)pitch2[firstPitchIndex] * (jj + 1) + firstPitchIndex * 512;
                    for (int i = start; i < end; i++)
                    {
                        if (i >= mono.Length) break;
                        if (mono[i] > max)
                        {
                            max = mono[i];
                            maxi = i;
                        }
                    }
                    jj++;
                }

                // Fill gaps in pitch2
                float last = 100;
                for (int i = 0; i < pitch2.Length; i++)
                {
                    if (pitch2[i] == -1 || pitch2[i] == 0) pitch2[i] = last;
                    else last = pitch2[i];
                    if (pitch2[i] < 20) pitch2[i] = 20;
                }

                // Build pitch markers
                if (maxi != -1)
                {
                    ptchpt.Add(maxi);
                    int lasti = maxi;

                    while (true)
                    {
                        float m = (lasti % 512) / 512f;
                        int p = lasti / 512;
                        if (p > 278) break;
                        int dist = (int)((1 - m) * pitch2[p] + m * pitch2[p + 1]);
                        int dist0 = dist * 6 / 5;
                        int dist1 = dist * 4 / 5;
                        if (dist <= 0) break;

                        max = 0;
                        maxi = lasti + dist;
                        for (int i = lasti + dist1; i < lasti + dist0; i++)
                        {
                            if (i >= mono.Length) break;
                            if (mono[i] > max)
                            {
                                max = mono[i];
                                maxi = i;
                            }
                        }
                        ptchpt.Add(maxi);
                        lasti = maxi;
                    }
                }

                if (ptchpt.Count < 2)
                {
                    SaveWav(mono, sampleRate, outputPath);
                    return;
                }

                // PSOLA synthesis at target frequency
                float[] output = new float[mono.Length];
                float pitchLength = sampleRate / freq;
                float waveletLength = pitchLength; // formant = 1
                float[] weights = new float[] { 1, 3, 8, 3, 1 };
                int[] weightOffsets = new int[] { -2, -1, 0, 1, 2 };

                for (float f = 0; f < output.Length; f += pitchLength)
                {
                    PlaceGranule(ref output, mono, ptchpt, waveletLength, f, f, weights, weightOffsets);
                }

                Normalize(ref output);
                SaveWav(output, sampleRate, outputPath);
            }
        }

        private static void PlaceGranule(ref float[] samples, float[] audio, List<int> pitchMarker,
            float waveletLength, float mid, float posIn, float[] weights, int[] weightOffsets)
        {
            for (int i = -((int)waveletLength + 1); i <= (int)waveletLength; i++)
            {
                int idx = (int)(mid + i);
                if (idx >= 0 && idx < samples.Length)
                {
                    float phase = i / waveletLength;
                    float vol = (float)Math.Pow(Math.Cos(phase * Math.PI / 2), 2);
                    phase -= 0.5f;
                    samples[idx] += vol * GetAudio(audio, pitchMarker, posIn, phase, weights, weightOffsets);
                }
            }
        }

        private static float GetAudio(float[] audio, List<int> pitchMarker, float pos, float positionInWavelet,
            float[] weights, int[] weightOffsets)
        {
            if (pitchMarker.Count < 2) return 0;
            pos += pitchMarker[0];

            if (pos >= 0 && pos < pitchMarker[pitchMarker.Count - 2] - pitchMarker[0])
            {
                float p = -1;
                for (int i = 0; i < pitchMarker.Count - 1; i++)
                {
                    if (pos >= pitchMarker[i] && pos < pitchMarker[i + 1])
                    {
                        p = i;
                        p += (pos - pitchMarker[i]) / (pitchMarker[i + 1] - pitchMarker[i]);
                    }
                }

                while (positionInWavelet < 0) { p -= 1; positionInWavelet++; }
                while (positionInWavelet > 1) { p += 1; positionInWavelet--; }

                float output = 0;
                for (int i = 0; i < weightOffsets.Length && i < weights.Length; i++)
                {
                    int cp = (int)p + weightOffsets[i];
                    if (cp >= 0 && cp < pitchMarker.Count - 1)
                    {
                        float pp = (1 - positionInWavelet) * pitchMarker[cp] + positionInWavelet * pitchMarker[cp + 1];
                        float pf = pp % 1;
                        if (pp >= 0 && pp <= audio.Length - 2)
                        {
                            output += weights[i] * ((1 - pf) * audio[(int)pp] + pf * audio[(int)pp + 1]);
                        }
                    }
                }
                return output;
            }
            return 0;
        }

        private static void Normalize(ref float[] f)
        {
            float max = 0;
            for (int i = 0; i < f.Length; i++)
            {
                if (Math.Abs(f[i]) > max) max = Math.Abs(f[i]);
            }
            if (max != 0)
            {
                for (int i = 0; i < f.Length; i++) f[i] /= max;
            }
        }

        private static int Autocorrelation(ref float[] input, int startPos, int searchSize, int window)
        {
            float[] array = new float[window];
            for (int i = 0; i < searchSize; i++)
            {
                array[i] = 0;
                for (int j = 0; j < window; j++)
                {
                    if (startPos + j + i < input.Length)
                        array[i] += input[startPos + j] * input[startPos + j + i];
                }
            }

            float maxMagn = 0;
            int L = -1;
            for (int i = 1; i < array.Length - 1; i++)
            {
                if (array[i] > array[i + 1] && array[i] >= array[i - 1])
                {
                    if (array[i] > maxMagn)
                    {
                        maxMagn = array[i];
                        L = i;
                    }
                }
            }
            return L;
        }

        private static void SaveWav(float[] output, int sampleRate, string fileName)
        {
            var waveFormat = new WaveFormat(sampleRate, 16, 1);
            using (var writer = new WaveFileWriter(fileName, waveFormat))
            {
                writer.WriteSamples(output, 0, output.Length);
            }
        }
    }
}
