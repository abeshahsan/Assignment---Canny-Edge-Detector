% ------------------------------------------------------------------------------
%
% This is a custom class for lab reports. It is based on the article class.
% It imports necessary packages, custom settings, styles, and commands.
% ------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mylabreport}[My custom lab report class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass{article}

% Package imports
\RequirePackage[a4paper, margin=0.75in]{geometry}
\RequirePackage[none]{hyphenat}
\RequirePackage{amsmath}
\RequirePackage{graphicx}

\RequirePackage{amssymb}
\RequirePackage{algpseudocode}
\RequirePackage{algorithm}
\RequirePackage{tikz}


\RequirePackage{wrapfig}
\RequirePackage{adjustbox}

\RequirePackage{subcaption}
\RequirePackage{multirow}
\RequirePackage{tabularx}

\RequirePackage{parskip}
\RequirePackage{setspace}
\RequirePackage{customstyles}
\RequirePackage[toc,page]{appendix}
\RequirePackage{multicol}


% End of package imports


% Custom commands

\newcommand{\labnumber}{LabNumber}
\newcommand{\setlabnumber}{\renewcommand{\labnumber}}

\newcommand{\coursecode}{CourseCode}
\newcommand{\setcoursecode}{\renewcommand{\coursecode}}

\newcommand{\coursetitle}{CourseTitle}
\newcommand{\setcoursetitle}{\renewcommand{\coursetitle}}

\newcommand{\labtitle}{LabTitle}
\newcommand{\setlabtitle}{\renewcommand{\labtitle}}

\newcommand{\studentid}{StudentID}
\newcommand{\setstudentid}{\renewcommand{\studentid}}

\newcommand{\studentname}{StudentName}
\newcommand{\setstudentname}{\renewcommand{\studentname}}

\newcommand{\labgroup}{LabGroup}
\newcommand{\setlabgroup}{\renewcommand{\labgroup}}

\newcommand{\student}{
    Author Name \\
    ID \\
    Dept \\
}
\newcommand{\setstudent}{\renewcommand{\student}}

% End of custom commands



% Page settings
\tolerance=1
\emergencystretch=\maxdimen
\hyphenpenalty=10000
\hbadness=10000
% End of page settings


% Custom styles

% Title page
\newcommand{\maketitlepage}{
    \begin{titlepage}
        \centering % Center everything on the page

        %----------------------------------------------------------------------------------------
        %	HEADING SECTIONS
        %----------------------------------------------------------------------------------------

        \includegraphics[scale=0.35]{iut_logo.png}\\[1cm]
        {\Large \textbf{Department of Computer Science and Engineering}\\[0.3cm]}
        {\large \textbf{Islamic University of Technology}\\[1.5cm] }
        {\Large \textbf{\coursecode:} \coursetitle\\[0.5cm]}

        %----------------------------------------------------------------------------------------
        %	TITLE SECTION
        %----------------------------------------------------------------------------------------

        {\huge \textbf{Assignment \labnumber}\\[0.2cm] }
        {\huge \textbf{\labtitle}\\[3cm]}

        %----------------------------------------------------------------------------------------
        %	AUTHOR SECTION
        %----------------------------------------------------------------------------------------
        
        {\Huge \student \vspace{4cm}}
        

        %----------------------------------------------------------------------------------------
        %	DATE SECTION
        %----------------------------------------------------------------------------------------

        {\Large \today}\\[2cm] % Date, change the \today to a set date if you want to be precise

        \vfill % Fill the rest of the page with whitespace

    \end{titlepage}
}

% Page header and footer
\RequirePackage{fancyhdr}
\fancyhf{}
\headheight=15pt

\fancypagestyle{firstpage}{
    \fancyhf{}
    \lhead{ID: \studentid}
    \rhead{Assignment \labnumber}
    \chead{\coursecode: \coursetitle}
    \cfoot{\thepage}
}

\pagestyle{plain}

\AtBeginDocument{
    \thispagestyle{firstpage}
}

\endinput