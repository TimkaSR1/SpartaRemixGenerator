using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NAudio.Wave;

namespace PitchCorrector297
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            DoubleBuffered = true;
            SO = new SoundOut();
            SO.Play();
        }
        // extract 2.5
        private void button4_Click(object sender, EventArgs e)
        {

            if (AFR != null)
            {
                if (SO.providers.Contains(AFR)) SO.providers.Remove(AFR);
                long origPos = AFR.Position;
                AFR.Position = ((long)(SelectedPosition * AFR.WaveFormat.AverageBytesPerSecond) / 32) * 32;

                if (AFR.WaveFormat.Channels == 2) STM = AFR.ToMono();
                else STM = AFR;

                ExtractedSamples = new float[(int)(AFR.WaveFormat.SampleRate * 2.5)];

                STM.Read(ExtractedSamples, 0, ExtractedSamples.Length);
                STM = null;

                AFR.Position = origPos;

                Granularize();
            }
        }
        // save dialog
        private void button7_Click(object sender, EventArgs e)
        {
            saveFileDialog1.ShowDialog();
        }
        // play marked
        private void button5_Click(object sender, EventArgs e)
        {
            float[] f = ExtractMarked();

            if (SO.providers.Contains(FAS)) SO.providers.Remove(FAS);
            FAS = new FloatArraySource(f, AFR.WaveFormat.SampleRate);
            SO.providers.Add(FAS);
        }
        // reload
        private void button6_Click(object sender, EventArgs e)
        {
            ExtractWithSettings();
        }
        private void openFileDialog2_FileOk(object sender, CancelEventArgs e)
        {
            AFR = new AudioFileReader(openFileDialog2.FileName);
            file = openFileDialog2.FileName;

            ExtractedSamples = new float[(int)(AFR.WaveFormat.SampleRate * 2.5f)];

            if (AFR.WaveFormat.Channels == 2) STM = AFR.ToMono();
            else STM = AFR;

            STM.Read(ExtractedSamples, 0, ExtractedSamples.Length);
            STM = null;

            Granularize();
        }    

        // Red marker rectangle
        int TimeRStart = 200;
        int TimeREnd = 300;
        private void panel2_Paint(object sender, PaintEventArgs e)
        {
            if (ExtractedSamples != null)
            {
                e.Graphics.FillRectangle(Brushes.Red, TimeRStart, 0, TimeREnd - TimeRStart, panel2.Height);
                float m = 0;

                for (int i = 0; i < ExtractedSamples.Length; i++)
                {
                    if (Math.Abs(ExtractedSamples[i]) > m)
                    {
                        m = Math.Abs(ExtractedSamples[i]);
                    }
                }
                if (m != 0)
                {
                    for (int i = 0; i < ExtractedSamples.Length; i++)
                    {
                        ExtractedSamples[i] /= m;
                    }

                    for (int i = 0; i < 1000; i++)
                    {
                        float min = 999f;
                        float max = -999f;

                        for (int j = (int)((ExtractedSamples.Length / 1000f) * i); j < (ExtractedSamples.Length / 1000f) * (i + 1); j++)
                        {
                            if (j < ExtractedSamples.Length)
                            {
                                if (ExtractedSamples[j] > max)
                                {
                                    max = ExtractedSamples[j];
                                }
                                if (ExtractedSamples[j] < min)
                                {
                                    min = ExtractedSamples[j];
                                }
                            }
                        }

                        int p0 = (int)((((min) + 1) / 2) * panel2.Height);
                        int p1 = (int)((((max) + 1) / 2) * panel2.Height);
                        e.Graphics.DrawLine(Pens.Green, i, p0, i, p1);
                    }
                }
            }
        }
        private void panel2_MouseDown(object sender, MouseEventArgs e)
        {
            TimeRStart = panel2.PointToClient(MousePosition).X;
        }
        private void panel2_MouseUp(object sender, MouseEventArgs e)
        {
            TimeREnd = panel2.PointToClient(MousePosition).X;

            if (TimeREnd < TimeRStart)
            {
                int pom = TimeRStart;
                TimeRStart = TimeREnd;
                TimeREnd = pom;
            }

            if (TimeRStart < 0) TimeRStart = 0;
            if (TimeREnd >= 1000) TimeREnd = 999;

            panel2.Invalidate();
        }

        // SOUND STORAGE
        PitchMarkerSound pms = null;
        float SelectedPosition = 0;
        FloatArraySource FAS = null;
        AudioFileReader AFR = null;
        ISampleProvider STM = null;
        SoundOut SO;
        float[] ExtractedSamples;
        string file;

        // INNER WORKINGS
        private void saveFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
            float[] f = ExtractMarked();

            WaveFormat waveFormat = new WaveFormat(AFR.WaveFormat.SampleRate, 16, 1);
            using (WaveFileWriter writer = new WaveFileWriter(saveFileDialog1.FileName, waveFormat))
            {
                writer.WriteSamples(f, 0, f.Length);
            }
        }
        public void Granularize()
        {
            // AUDIO holds 2.5 seconds of audio from the video source at user selected time.

            int[] pitch = new int[280];

            // gets autocorelation of AUDIO against itself to get estimated pitch at various time points spread by 512 samples.
            for (int i = 0; i < 280; i++)
            {
                pitch[i] = Autocorelation(ref ExtractedSamples, 512 * i, 512, 2048);
            }

            // try to detect and fix "octave" errors.

            float[] pitch2 = new float[280];

            for (int i = 0; i < 280; i++)
            {
                List<int> pp = new List<int>();
                for (int j = -7 / 2; j < 7 / 2 + 1; j++) if (j + i >= 0 && j + i < 280) if (pitch[j + i] > -1) pp.Add(pitch[j + i]);
                if (pp.Count == 0) { pitch2[i] = -1; continue; }
                pp.Sort();

                int ppp = pp[pp.Count / 2];
                List<int> cand = new List<int>();

                int newp = -1;
                int minDiff = int.MaxValue;

                for (int j = 1; j < 5; j++)
                {
                    for (int k = 1; k < 5; k++)
                    {
                        if (Math.Abs(ppp - pitch[i] * j / k) < minDiff)
                        {
                            newp = pitch[i] * j / k;
                            minDiff = Math.Abs(ppp - pitch[i] * j / k);
                        }
                    }
                }

                pitch2[i] = newp;
                if (newp == 0) pitch2[i] = -1;
            }

            // pitch2 is now used to find "pitch points" (ptchpt) requered for "PSOLA"

            int firstPitchIndex = -1;
            for (int i = 0; i < pitch2.Length; i++)
            {
                if (pitch2[i] > -1)
                {
                    firstPitchIndex = i;
                    break;
                }
            }

            if (firstPitchIndex == -1) return;

                List<int> ptchpt = new List<int>();

            float max = 0;
            int maxi = -1;

            int jj = 0;
            while (maxi == -1)
            {

                for (int i = firstPitchIndex * 512 + jj * (int)pitch2[firstPitchIndex]; i < pitch2[firstPitchIndex] * (jj+1) + firstPitchIndex * 512; i++)
                {
                    if (i > ExtractedSamples.Length) break;
                    if (ExtractedSamples[i] > max)
                    {
                        max = ExtractedSamples[i];
                        maxi = i;
                    }
                }
                jj++;
            }

            float last = 100;
            for (int i = 0; i < pitch2.Length; i++)
            {
                if (pitch2[i] == -1 || pitch2[i] == 0) pitch2[i] = last;
                else last = pitch2[i];

                if (pitch2[i] < 20) pitch2[i] = 20;
            }

            // assume max deviation from detected pitch by 1/5

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
                        if (i >= ExtractedSamples.Length) break;
                        if (ExtractedSamples[i] > max)
                        {
                            max = ExtractedSamples[i];
                            maxi = i;
                        }

                    }
                    ptchpt.Add(maxi);
                    lasti = maxi;
                }
            }

            // now that we have peaks, we can declare PSOLA-ready sample. Here it will be used to just output single sound.
            // in my future DAW, called Sparta Remix Studio, it will produce real time sound by firing granules.

            pms = new PitchMarkerSound(ExtractedSamples, ptchpt);



            // Now "extractedSamples" will hold corrected sound, while pms holds original.
            ExtractWithSettings();
        }
        public void ExtractWithSettings()
        {
            if (pms != null)
            {
                ExtractedSamples = new float[(int)(AFR.WaveFormat.SampleRate * 2.5f)];

                float freq = 220;
                float fmt = 1;
                float spd = 1;
                float[] svt = new float[] { 1, 3, 8, 3, 1 };
                int[] ind = new int[] { -2, -1, 0, 1, 2 };

                try
                {
                    freq = float.Parse(textBox1.Text, CultureInfo.InvariantCulture);
                    spd = float.Parse(textBox2.Text, CultureInfo.InvariantCulture);
                    fmt = float.Parse(textBox4.Text, CultureInfo.InvariantCulture);
                    string[] s = textBox3.Text.Split(' ');
                    svt = new float[s.Length];
                    ind = new int[s.Length];

                    for (int i = 0; i < s.Length; i++) ind[i] = i - (s.Length / 2);
                    for (int i = 0; i < s.Length; i++) svt[i] = float.Parse(s[i], CultureInfo.InvariantCulture);
                }
                catch
                {

                }


                float pitchLenght = AFR.WaveFormat.SampleRate / freq;
                float waveletLenght = pitchLenght / fmt;
                pms.WeightOffsets = ind;
                pms.Weights = svt;

                if (pitchLenght > 0 && waveletLenght < 99999)
                {
                    for (float f = 0; f < ExtractedSamples.Length; f += pitchLenght)
                    {
                        pms.PlaceGranule(ref ExtractedSamples, waveletLenght, f, spd * f);
                    }
                }

                Normalize(ref ExtractedSamples);

                panel2.Invalidate();
            }
        }
        float[] ExtractMarked()
        {
            float[] f = new float[(ExtractedSamples.Length / 1000) * (TimeREnd - TimeRStart)];
            int s = (int)((TimeRStart / 1000f) * ExtractedSamples.Length);
            for (int i = 0; i < f.Length; i++)
            {
                if (s + i < ExtractedSamples.Length) f[i] = ExtractedSamples[s + i];
            }
            return f;
        }

        // GENERAL AUDIO-RELATED ALGORHITMS
        public void Normalize(ref float[] f)
        {
                float max = 0;
                for (int i = 0; i < f.Length; i++)
                {
                    if (Math.Abs(f[i]) > max)
                    {
                        max = Math.Abs(f[i]);
                    }
                }
                if (max != 0) for (int i = 0; i < f.Length; i++) f[i] /= max;
            
        }
        public void SaveAudio(float[] output, int sr, string fileName, bool normalize)
        {
            FileStream fs = new FileStream(fileName, FileMode.Create);

            fs.WriteByte(0x52);
            fs.WriteByte(0x49);
            fs.WriteByte(0x46);
            fs.WriteByte(0x46);

            int l = 24 + 8 + output.Length;
            fs.WriteByte((byte)((l >> 0) % 256));
            fs.WriteByte((byte)((l >> 8) % 256));
            fs.WriteByte((byte)((l >> 16) % 256));
            fs.WriteByte((byte)((l >> 24) % 256));


            fs.WriteByte(0x57);
            fs.WriteByte(0x41);
            fs.WriteByte(0x56);
            fs.WriteByte(0x45);

            fs.WriteByte(0x66);
            fs.WriteByte(0x6d);
            fs.WriteByte(0x74);
            fs.WriteByte(0x20);

            fs.WriteByte(0x10);
            fs.WriteByte(0x00);
            fs.WriteByte(0x00);
            fs.WriteByte(0x00);

            fs.WriteByte(0x01);
            fs.WriteByte(0x00);
            fs.WriteByte(0x01);
            fs.WriteByte(0x00);

            l = sr;
            fs.WriteByte((byte)((l >> 0) % 256));
            fs.WriteByte((byte)((l >> 8) % 256));
            fs.WriteByte((byte)((l >> 16) % 256));
            fs.WriteByte((byte)((l >> 24) % 256));

            l = sr * 2;
            fs.WriteByte((byte)((l >> 0) % 256));
            fs.WriteByte((byte)((l >> 8) % 256));
            fs.WriteByte((byte)((l >> 16) % 256));
            fs.WriteByte((byte)((l >> 24) % 256));

            fs.WriteByte(0x02);
            fs.WriteByte(0x00);
            fs.WriteByte(0x10);
            fs.WriteByte(0x00);

            fs.WriteByte(0x64);
            fs.WriteByte(0x61);
            fs.WriteByte(0x74);
            fs.WriteByte(0x61);

            l = output.Length * 2;
            fs.WriteByte((byte)((l >> 0) % 256));
            fs.WriteByte((byte)((l >> 8) % 256));
            fs.WriteByte((byte)((l >> 16) % 256));
            fs.WriteByte((byte)((l >> 24) % 256));

            

            for (int i = 0; i < output.Length; i++)
            {
                short s = (short)(output[i] * 32767);
                fs.WriteByte((byte)((s >> 0) % 256));
                fs.WriteByte((byte)((s >> 8) % 256));
                //fs.WriteByte((byte)((s >> 0) % 256));
            }

            fs.Close();
            fs.Dispose();
        }        
        public static int Autocorelation(ref float[] input, int startPos, int searchSize, int window)
        {
            float[] array = new float[window];

            for (int i = 0; i < searchSize; i++)
            {
                array[i] = 0;

                for (int j = 0; j < window; j++)
                {
                    if (startPos + j + i < input.Length) array[i] += input[startPos + j] * input[startPos + j + i];
                }
            }

            float maxMagn = 0;
            int L = -1;

            for (int i = 1; i < array.Length - 1; i++)
            {
                if (array[i] > array[i + 1] && array[i] >= array[i - 1]) if (array[i] > maxMagn)
                    {
                        maxMagn = array[i];
                        L = i;
                    }
            }

            return L;
        }
        public static float[] HighPass6(float[] input, float frequency, float sampleRate)
        {
            float RC = (float)(1f / (2 * Math.PI * frequency));
            float alpha = RC / ((1 / sampleRate) + RC);


            float[] output = new float[input.Length];
            output[0] = input[0];
            for (int i = 1; i < input.Length; i++) output[i] = alpha * (output[i - 1] + input[i] - input[i - 1]);

            return output;
        }
        public static float[] LowPass6(float[] input, float frequency, float sampleRate)
        {
            float RC = (float)(1f / (2 * Math.PI * frequency));
            float alpha = RC / ((1 / sampleRate) + RC);


            float[] output = new float[input.Length];
            output[0] = input[0];
            for (int i = 1; i < input.Length; i++) output[i] = output[i - 1] + alpha * (input[i] - output[i - 1]);

            return output;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            openFileDialog2.ShowDialog();
        }

        
    }

    public class SoundOut : ISampleProvider, IDisposable
    {
        public List<ISampleProvider> providers;
        public WaveOut outputDevice;
        public int pos = 0;

        public SoundOut()
        {
            SampleRate = 48000;
            outputDevice = new WaveOut();
            outputDevice.Init(this, true);
            providers = new List<ISampleProvider>();
        }

        public void Play()
        {
            outputDevice.Play();
        }

        public void Stop()
        {
            outputDevice.Stop();
        }

        public void Dispose()
        {
            outputDevice.Stop();
            outputDevice.Dispose();
        }

        public int SampleRate { get; set; }

        public WaveFormat WaveFormat
        {
            get
            {
                return WaveFormat.CreateIeeeFloatWaveFormat(SampleRate, 2);
            }
        }
        public int Read(float[] buffer, int offset, int count)
        {
            float[] secodaryBuffer = new float[count];
            Array.Clear(buffer, 0, buffer.Length);
            foreach (ISampleProvider provider in providers)
            {
                Array.Clear(secodaryBuffer, 0, count);
                int r = 0;
                if (provider.WaveFormat.Channels == 2) r = provider.Read(secodaryBuffer, offset, count);
                else r = provider.Read(secodaryBuffer, offset, count / 2);
                if (provider.WaveFormat.Channels == 2) for (int i = 0; i < r; i++) buffer[i] += secodaryBuffer[i];
                else for (int i = 0; i < r; i++) { buffer[2 * i] += secodaryBuffer[i]; buffer[2 * i + 1] += secodaryBuffer[i]; }
            }

            pos += count;
            return count;
        }
    }
    public class FloatArraySource : ISampleProvider
    {
        private float[] samples;
        private long pos;
        private int sampleRate;

        public FloatArraySource(float[] samples, int sr)
        {
            this.samples = samples;
            sampleRate = sr;
        }

        public WaveFormat WaveFormat
        {
            get
            {
                return WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            }
        }
        public int Read(float[] buffer, int offset, int count)
        {
            int read = 0;
            for (int i = 0; i < count; i++)
            {
                if ((pos + i) < samples.Length)
                {
                    buffer[i] = samples[pos + i];
                    read++;
                }
            }
            pos += count;
            return read;
        }
    }
    public class PitchMarkerSound
    {
        float[] audio;
        List<int> pitchMarker;

        public float[] Weights
        {
            get; set;
        }
        public int[] WeightOffsets
        {
            get; set;
        }

        public float GetAudio(float pos, float positionInWavelet)
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
                        p += ((pos - pitchMarker[i]) / (pitchMarker[i + 1] - pitchMarker[i]));
                    }
                }

                while (positionInWavelet < 0) { p -= 1; positionInWavelet++; }
                while (positionInWavelet > 1) { p += 1; positionInWavelet--; }

                float output = 0;

                for (int i = 0; i < WeightOffsets.Length && i < Weights.Length; i++)
                {
                    int cp = (int)p + WeightOffsets[i];

                    if (cp >= 0 && cp < pitchMarker.Count - 1)
                    {
                        float pp = (1 - positionInWavelet) * pitchMarker[cp] + positionInWavelet * pitchMarker[cp + 1];
                        float pf = pp % 1;

                        if (pp >= 0 && pp <= audio.Length - 2) output += (1/* - p % 1f*/) * ((1 - pf) * audio[(int)pp] + pf * audio[(int)pp + 1]);

                        //if (cp < pitchMarker.Count - 2) output += (p % 1f) * ((1 - pf) * audio[(int)pp + 1] + pf * audio[(int)pp + 2]);
                    }
                }

                return output;
            }
            else return 0;
        }

        public PitchMarkerSound(float[] audio, List<int> pitchMarker)
        {
            this.audio = audio;
            this.pitchMarker = pitchMarker;

            Weights = new float[] { 1 };
            WeightOffsets = new int[] { 0 };
        }

        public void PlaceGranule(ref float[] samples, float waveletLenght, float mid, float posIn)
        {
            for (int i = -((int)waveletLenght + 1); i <= (int)(waveletLenght); i++)
            {
                if (mid + i >= 0 && mid + i < samples.Length)
                {
                    float phase = i / waveletLenght;
                    float vol = (float)Math.Pow(Math.Cos(phase * Math.PI / 2), 2);

                    phase -= 1 / 2f;
                    samples[(int)(mid + i)] += vol * GetAudio(posIn, phase);
                }
            }
        }
    }

}
