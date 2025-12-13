using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace PitchCorrector297
{
    static class Program
    {
        /// <summary>
        /// Main entry point. Supports CLI mode when args provided.
        /// </summary>
        [STAThread]
        static int Main(string[] args)
        {
            // CLI mode: PitchCorrector297.exe <input.wav> <output.wav> [frequency]
            if (args.Length >= 2)
            {
                return PitchCorrectorCLI.Run(args);
            }

            // GUI mode
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
            return 0;
        }
    }
}
