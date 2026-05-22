# 第1章 教程

本章是对 Go 基本组件的巡礼。我们希望提供足够的信息和示例，让你能够快速上手并开始做些有用的事情。这里的例子（以及整本书中的例子）都是针对你在现实世界中可能遇到的任务。在本章中，我们将让你尝鲜 Go 所能编写的各种程序——从简单的文件处理和一点图形处理，到并发的互联网客户端和服务器。我们当然不会在第一章中解释所有内容，但用一种新语言研究这样的程序可能是入门的一种有效方法。

当你学习一门新语言时，有一种自然的倾向，会按照你已经掌握的语言的方式来编写代码。在学习 Go 时要注意这种偏见并尽量避免它。我们试图说明和解释如何编写好的 Go 代码，因此在编写自己的代码时，请将这里的代码作为指南。

## 1.1. Hello, World

我们从如今已成为传统的"hello, world"示例开始，它出现在 1978 年出版的《C 程序设计语言》一书的开头。C 语言是 Go 最直接的影响之一，而"hello, world"展示了许多核心思想。

```
gopl.io/ch1/helloworld

package main

import "fmt"

func main() {
    fmt.Println("Hello, BF")
}
```

Go 是一种编译型语言。Go 工具链将源程序及其依赖项转换成计算机本机机器语言的指令。这些工具通过一个名为 `go` 的单一命令来访问，该命令有许多子命令。其中最简单的子命令是 `run`，它编译一个或多个以 `.go` 结尾的源文件，将其与库链接，然后运行生成的可执行文件。（本书通篇使用 `$` 作为命令提示符。）

```
$ go run helloworld.go
```

不出所料，这会输出：

```
Hello, BF
```

Go 原生支持 Unicode，因此它可以处理世界上所有语言的文本。

如果程序不仅仅是一次性实验，你可能希望编译一次并保存编译结果以备后用。这可以通过 `go build` 来完成：

```
$ go build helloworld.go
```

这会创建一个名为 `helloworld` 的可执行二进制文件，可以随时运行，无需进一步处理：

```
$ ./helloworld
Hello, BF
```

我们在每个重要示例上都做了标签，提醒你可以从本书的源代码仓库 `gopl.io` 获取代码：

```
gopl.io/ch1/helloworld
```

如果你运行 `go get gopl.io/ch1/helloworld`，它会获取源代码并将其放入相应的目录中。关于此主题的更多内容见第 2.6 节和第 10.7 节。

现在让我们讨论一下程序本身。Go 代码被组织成**包**（package），这类似于其他语言中的库或模块。一个包由一个目录中的一个或多个 `.go` 源文件组成，这些文件定义了该包的功能。每个源文件以 `package` 声明开头（这里为 `package main`），说明该文件属于哪个包，然后是它导入的其他包的列表，接着是该文件中存储的程序声明。

Go 标准库有超过 100 个包，用于常见任务，如输入输出、排序和文本处理。例如，`fmt` 包包含用于格式化输出和扫描输入的函数。`Println` 是 `fmt` 中基本的输出函数之一；它打印一个或多个值，用空格分隔，末尾带有一个换行符，这样这些值就以单行输出的形式出现。

`package main` 是特殊的。它定义了一个独立的可执行程序，而不是一个库。在 `package main` 中，`main` 函数也是特殊的——它是程序执行开始的地方。`main` 做什么，程序就做什么。当然，`main` 通常会调用其他包中的函数来完成大部分工作，例如调用 `fmt.Println` 函数。

我们必须告诉编译器这个源文件需要哪些包；这就是在 `package` 声明之后的 `import` 声明的作用。"hello, world" 程序只使用了来自另一个包的一个函数，但大多数程序会导入更多的包。

你必须精确地导入你需要的包。如果有缺失的导入或不必要的导入，程序将无法编译。这种严格的要求防止了未使用的包随着程序的演进而积累。

`import` 声明必须跟在 `package` 声明之后。之后，程序由函数、变量、常量和类型的声明（分别由关键字 `func`、`var`、`const` 和 `type` 引入）组成；在大多数情况下，声明的顺序并不重要。这个程序是尽可能简短的，因为它只声明了一个函数，而这个函数又只调用了另一个函数。为了节省篇幅，我们在展示示例时有时不会显示 `package` 和 `import` 声明，但它们在源文件中是存在的，并且必须存在才能编译代码。

函数声明由关键字 `func`、函数名、参数列表（`main` 为空）、结果列表（这里也为空）和函数体——定义其功能的语句——组成，用花括号括起来。我们将在第 5 章更详细地介绍函数。

Go 不要求在语句或声明的末尾加分号，除非两个或更多语句出现在同一行。实际上，跟在某些标记后面的换行符会被转换为分号，因此换行符的放置位置对于 Go 代码的正确解析很重要。例如，函数的左花括号 `{` 必须与 `func` 声明的末尾在同一行，而不能单独一行；在表达式 `x + y` 中，换行符允许在 `+` 之后，但不允许在之前。

Go 对代码格式有强硬立场。`gofmt` 工具将代码重写为标准格式，`go` 工具的 `fmt` 子命令将 `gofmt` 应用于指定包中的所有文件，或者默认应用于当前目录中的所有文件。本书中的所有 Go 源文件都经过 `gofmt` 处理，你应该养成对自己的代码也这样做的习惯。通过强制规定标准格式，消除了大量关于琐事无意义的争论，更重要的是，它使得各种自动化的源代码转换成为可能，如果允许任意格式化，这些转换将无法实现。

许多文本编辑器可以配置为在每次保存文件时运行 `gofmt`，这样你的源代码始终格式正确。一个相关的工具 `goimports` 还可以根据需要管理导入声明的插入和删除。它不是标准发行版的一部分，但你可以通过以下命令获取它：

```
$ go get golang.org/x/tools/cmd/goimports
```

对于大多数用户来说，下载和构建包、运行测试、显示文档等的常用方式是通过 `go` 工具，我们将在第 10.7 节中介绍。

## 1.2. 命令行参数

大多数程序处理一些输入以产生一些输出；这基本上就是计算的定义。但是程序如何获取输入呢？一些程序生成数据，但通常输入来自以下外部来源：
- 文件
- 命令行参数
- 环境变量
- 来自网络连接的 socket
- 网页
- 键盘或鼠标等设备
- API 调用

命令行参数是一种常见的方式，尤其是在批处理或后台使用中，可以向程序提供微调或控制信息。

`os` 包提供了一个名为 `Args` 的字符串切片（`string` 切片），它是跨平台的，作为 `os` 包的外部接口的一部分。让我们快速编写一个使用 `os.Args` 的程序；以下是 `echo` 的一个实现，它将其命令行参数打印在一行上：

```
gopl.io/ch1/echo1

// Echo1 prints its command-line arguments.
package main

import (
    "fmt"
    "os"
)

func main() {
    var s, sep string
    for i := 1; i < len(os.Args); i++ {
        s += sep + os.Args[i]
        sep = " "
    }
    fmt.Println(s)
}
```

注释以 `//` 开头。按照惯例，在每个包声明之前的注释是包的文档注释（第 10.7.4 节）。在 `main` 之前的注释是对整个程序的评论，尽管对于像这样的小程序来说，它可能只是一个简单的说明。

变量声明采用以下形式：

```
var s, sep string
```

声明了两个 `string` 类型的变量 `s` 和 `sep`。我们稍后会讨论变量声明；目前，先相信这个说法。

变量 `s` 的初始值是通过重复追加参数来构建结果的字符串。变量 `sep` 用作分隔符，在每次追加参数时，除了最后一个参数之外，都在参数前添加一个空格。

`for` 循环是 Go 中唯一的循环语句。它有多种形式：

```
for initialization; condition; post {
    // 零条或多条语句
}
```

大括号是必需的，左花括号必须与 `post` 语句在同一行。`for` 循环的三个部分都可以省略。如果省略了 `initialization` 和 `post`，分号也可以省略：

```
// 传统的 "while" 循环
for condition {
    // ...
}
```

如果条件被完全省略，则成为无限循环：

```
// 传统的无限循环
for {
    // ...
}
```

`echo` 程序还可以通过在一个循环中逐段打印其输出来实现，但这个版本通过重复向末尾追加新文本来构建一个字符串。`s` 的初始值为空（即值为 `""`），每次循环迭代都会添加一些文本；第一次迭代后还会插入一个空格，这样当循环结束时，每个参数之间就有一个空格。这是一个二次方复杂度的过程，如果参数数量很大可能会很昂贵，但对于 `echo` 来说，这不太可能。我们将在本章和下一章中展示 `echo` 的许多改进版本，这些版本将解决任何实际的低效问题。

循环索引变量 `i` 在 `for` 循环的第一部分中声明。`:=` 符号是**短变量声明**的一部分，这是一种声明一个或多个变量并根据初始化器值为其指定适当类型的语句；关于此内容的更多介绍在下一章。

递增语句 `i++` 将 1 加到 `i` 上；它等价于 `i += 1`，而后者又等价于 `i = i + 1`。还有一个对应的递减语句 `i--`，它减去 1。这些是语句，而不是像 C 语言家族大多数语言中的表达式，因此 `j = i++` 是非法的，并且它们只是后缀形式，所以 `--i` 也是非法的。

`for` 循环是 Go 中唯一的循环语句。它有多种形式，其中一种如下所示：

```
for initialization; condition; post {
    // 零条或多条语句
}
```

括号从不用在 `for` 循环的三个组件周围。然而，大括号是必需的，并且左花括号必须与 `post` 语句在同一行。

可选的初始化语句在循环开始之前执行。如果存在，它必须是一个简单语句，即短变量声明、递增或赋值语句，或函数调用。条件是一个布尔表达式，在每次循环迭代开始时求值；如果为 true，则执行循环控制的语句。`post` 语句在循环体之后执行，然后再次计算条件。当条件变为 false 时循环结束。

这些部分中的任何一个都可以省略。如果没有初始化和 post 语句，分号也可以省略：

```
// 传统的 "while" 循环
for condition {
    // ...
}
```

如果这些形式中的任何一种完全省略了条件，例如：

```
// 传统的无限循环
for {
    // ...
}
```

则循环是无限的，尽管这种形式的循环可以通过其他方式终止，如 `break` 或 `return` 语句。

`for` 循环的另一种形式遍历来自字符串或切片等数据类型的值的**范围**。为了说明这一点，这里是 `echo` 的第二个版本：

```
gopl.io/ch1/echo2

// Echo2 prints its command line arguments.
package main

import (
    "fmt"
    "os"
)

func main() {
    s, sep := "", ""
    for _, arg := range os.Args[1:] {
        s += sep + arg
        sep = " "
    }
    fmt.Println(s)
}
```

在循环的每次迭代中，`range` 产生一对值：索引和该索引处元素的值。在这个例子中，我们不需要索引，但 `range` 循环的语法要求如果我们处理元素，也必须处理索引。一种想法是将索引赋给一个明显是临时变量的变量如 `temp`，并忽略其值，但 Go 不允许未使用的局部变量，因此这会导致编译错误。

解决方案是使用**空标识符**，其名称为 `_`（即下划线）。无论何时语法需要变量名但程序逻辑不需要时，都可以使用空标识符，例如在只需要元素值时丢弃不需要的循环索引。大多数 Go 程序员可能会像上面那样使用 `range` 和 `_` 来编写 `echo` 程序，因为对 `os.Args` 的索引是隐式的，而不是显式的，因此更容易正确。

这个版本的程序使用短变量声明来声明和初始化 `s` 和 `sep`，但我们也可以分别声明这些变量。有几种声明字符串变量的方法；这些是等价的：

```
s := ""
var s string
var s = ""
var s string = ""
```

为什么你应该更喜欢一种形式而不是另一种？第一种形式，短变量声明，是最紧凑的，但只能用在函数内部，而不能用在包级别。第二种形式依赖于字符串的零初始化（在第 2.3 节讨论）。第三种形式很少使用。第四种形式在需要明确类型时有用。在实践中，你应该使用 `s := ""`，但前提是你明确想要一个空字符串；如果函数返回一个字符串值，你应该使用短变量声明。

现在让我们继续下一个示例，它在一个目录中的所有文件里查找重复的行。

`s` 的加法操作每次都会创建一个新的字符串，这在处理大量数据时效率低下。更有效的方法是使用 `strings.Join` 函数，我们在第 1.3 节中看到。但是目前，我们先使用了加法操作，因为它更容易理解。

`echo` 的这个版本使用 `range` 来迭代 `os.Args[1:]` 中的元素。我们不能直接在 `os.Args` 上使用 `range`，因为那样会包括程序名称。切片表达式 `os.Args[1:]` 表示从索引 1 到末尾的所有元素。切片表达式的范围是从起始索引到结束索引（但不包括），或者如果省略了一个或两个索引，则使用默认值。

还有一种方法可以不使用 `range` 来编写 `echo`：

```
gopl.io/ch1/echo3

func main() {
    fmt.Println(strings.Join(os.Args[1:], " "))
}
```

`strings.Join` 函数将切片中的元素连接成一个字符串，元素之间用提供的分隔符分隔。这是 Go 标准库中众多便利函数之一。

现在让我们看一个更有趣的程序，它查找重复的行。

## 1.3. 查找重复的行

处理文件数据、从网络读取数据、对数据进行排序、计算频率等是常见任务。Go 有丰富的标准库，可以高效地完成这些任务。

第一个程序 `dup` 从标准输入读取行，并打印重复的行及其出现的次数。程序使用一个 map 来存储每行出现的次数。

```
gopl.io/ch1/dup1

// Dup1 prints the text of each line that appears more than
// once in the standard input, preceded by its count.
package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    counts := make(map[string]int)
    input := bufio.NewScanner(os.Stdin)
    for input.Scan() {
        counts[input.Text()]++
    }
    // NOTE: ignoring potential errors from input.Err()
    for line, n := range counts {
        if n > 1 {
            fmt.Printf("%d\t%s\n", n, line)
        }
    }
}
```

和 `for` 一样，`if` 语句的条件周围从不使用括号，但大括号是必需的。可以有一个可选的 `else` 部分，在条件为 false 时执行。

map 保存一组键/值对，并提供常数时间的操作来存储、检索或测试集合中的项。键可以是任何其值可以用 `==` 比较的类型，字符串是最常见的例子；值可以是任何类型。在这个例子中，键是字符串，值是整数。内置函数 `make` 创建一个新的空 map；它还有其他用途。map 在第 4.3 节中详细讨论。

每次 `dup` 读取一行输入时，该行被用作 map 中的键，对应的值递增。语句 `counts[input.Text()]++` 等价于以下两条语句：

```
line := input.Text()
counts[line] = counts[line] + 1
```

如果 map 中尚不包含该键，也没有问题。第一次看到新行时，右边的表达式 `counts[line]` 的值为其类型的零值，对于 `int` 来说就是 0。

为了打印结果，我们使用另一个基于 `range` 的 `for` 循环，这次遍历 `counts` map。和之前一样，每次迭代产生两个结果：一个键和该键对应 map 元素的值。map 迭代的顺序未指定，但在实践中是随机的，每次运行都可能不同。这种设计是有意的，因为它防止程序依赖于任何没有保证的特定顺序。

接下来是 `bufio` 包，它有助于使输入输出高效便捷。它最有用的特性之一是一个名为 `Scanner` 的类型，它读取输入并将其分解为行或单词；它通常是处理自然按行输入的输入的最简单方式。

程序使用短变量声明创建一个新变量 `input`，它引用一个 `bufio.Scanner`：

```
input := bufio.NewScanner(os.Stdin)
```

扫描器从程序的标准输入读取。每次调用 `input.Scan()` 读取下一行并移除末尾的换行符；结果可以通过调用 `input.Text()` 获取。`Scan` 函数在有行时返回 `true`，没有更多输入时返回 `false`。

函数 `fmt.Printf` 类似于 C 和其他语言中的 `printf`，它从表达式列表中生成格式化输出。它的第一个参数是一个格式字符串，指定后续参数应该如何格式化。每个参数的格式由**转换字符**决定，即跟在百分号后面的一个字母。例如，`%d` 以十进制表示法格式化整数操作数，`%s` 展开为字符串操作数的值。

`Printf` 有十几个这样的转换字符，Go 程序员称之为**动词**。以下表格远非完整规范，但说明了可用的许多特性：

| 动词 | 描述 |
|------|------|
| `%d` | 十进制整数 |
| `%x`, `%o`, `%b` | 十六进制、八进制、二进制整数 |
| `%f`, `%g`, `%e` | 浮点数：3.141593, 3.141592653589793, 3.141593e+00 |
| `%t` | 布尔值：true 或 false |
| `%c` | 符文（Unicode 码点） |
| `%s` | 字符串 |
| `%q` | 带引号的字符串 "abc" 或符文 'c' |
| `%v` | 任何值，以自然格式显示 |
| `%T` | 任何值的类型 |
| `%%` | 字面百分号（无操作数） |

`dup1` 的格式字符串还包含一个制表符 `\t` 和一个换行符 `\n`。字符串字面量可能包含这样的**转义序列**，用于表示不可见字符。`Printf` 不会自动添加换行符。按照惯例，以字母 `f` 结尾的格式化函数（如 `Printf`、`fmt.Fprintf`、`fmt.Sprintf` 等）的格式字符串应遵循 `printf` 的规则；不以 `f` 结尾的函数（如 `Println`）则不同。

`dup` 的下一个版本从命名文件读取输入，而不是从标准输入读取，如果未指定文件，则回退到标准输入：

```
gopl.io/ch1/dup2

// Dup2 prints the count and text of lines that appear more than once
// in the input. It reads from stdin or from a list of named files.
package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    counts := make(map[string]int)
    files := os.Args[1:]
    if len(files) == 0 {
        countLines(os.Stdin, counts)
    } else {
        for _, arg := range files {
            f, err := os.Open(arg)
            if err != nil {
                fmt.Fprintf(os.Stderr, "dup2: %v\n", err)
                continue
            }
            countLines(f, counts)
            f.Close()
        }
    }
    for line, n := range counts {
        if n > 1 {
            fmt.Printf("%d\t%s\n", n, line)
        }
    }
}

func countLines(f *os.File, counts map[string]int) {
    input := bufio.NewScanner(f)
    for input.Scan() {
        counts[input.Text()]++
    }
    // NOTE: ignoring potential errors from input.Err()
}
```

这个程序引入了 `os.Open` 函数，它返回一个文件句柄。`os.Open` 返回两个值。第一个是打开的文件（一个 `*os.File`），当读取器不再需要该文件时，由调用者关闭。第二个值是一个 Go 的内置错误类型 `error`。如果 err 等于内置值 `nil`，则打开成功。文件被读取，当文件读取完毕时，`Close` 关闭该文件，释放所有资源。如果 err 不是 `nil`，则出现了问题。在这种情况下，错误值描述了问题。我们的简单错误处理使用 `Fprintf` 在标准错误流上打印一条消息，并使用动词 `%v`，它以默认格式显示任何类型的值，然后 `dup` 继续处理下一个文件；`continue` 语句进入外围 `for` 循环的下一次迭代。

为了保持代码示例规模合理，我们的早期示例有意在错误处理上有些粗心。显然我们必须检查 `os.Open` 的错误；然而，我们忽略了用 `input.Scan` 读取文件时不太可能发生的错误。我们会在跳过错误检查的地方做标记，并在第 5.4 节中详细介绍错误处理。

注意，对 `countLines` 的调用在其声明之前。函数和其他包级别实体可以以任何顺序声明。

map 是对 `make` 创建的数据结构的引用。当 map 传递给函数时，函数接收引用的副本，因此被调用函数对底层数据结构的任何更改也会通过调用者的 map 引用可见。在我们的示例中，`countLines` 插入到 `counts` map 中的值在 `main` 中也是可见的。

以上版本的 `dup` 以"流"模式运行，其中输入被按需读取并分解为行，因此原则上这些程序可以处理任意数量的输入。另一种方法是一次性将整个输入读入内存，然后一次性将其分割成行，再处理这些行。以下版本 `dup3` 以这种方式运行。它引入了 `ReadFile` 函数（来自 `io/ioutil` 包），该函数读取整个命名文件的内容，以及 `strings.Split`，它将字符串分割成子字符串切片。（`Split` 是 `strings.Join` 的反操作，我们在之前看到过。）

我们稍微简化了 `dup3`。首先，它只读取命名文件，而不是标准输入，因为 `ReadFile` 需要一个文件名参数。其次，我们将行计数移回 `main`，因为它现在只需要在一个地方。

```
gopl.io/ch1/dup3

package main

import (
    "fmt"
    "io/ioutil"
    "os"
    "strings"
)

func main() {
    counts := make(map[string]int)
    for _, filename := range os.Args[1:] {
        data, err := ioutil.ReadFile(filename)
        if err != nil {
            fmt.Fprintf(os.Stderr, "dup3: %v\n", err)
            continue
        }
        for _, line := range strings.Split(string(data), "\n") {
            counts[line]++
        }
    }
    for line, n := range counts {
        if n > 1 {
            fmt.Printf("%d\t%s\n", n, line)
        }
    }
}
```

`ReadFile` 返回一个字节切片，必须将其转换为字符串才能被 `strings.Split` 分割。我们将在第 3.5.4 节详细讨论字符串和字节切片。

在底层，`bufio.Scanner`、`ioutil.ReadFile` 和 `ioutil.WriteFile` 使用 `*os.File` 的 `Read` 和 `Write` 方法，但大多数程序员很少需要直接访问这些较低层的例程。像来自 `bufio` 和 `io/ioutil` 的这些更高级的函数更容易使用。

**练习 1.4**：修改 `dup2`，使其为每个出现重复行的文件打印该重复行所在的所有文件的名称。

## 1.4. 动画 GIF

下一个程序演示了 Go 标准图像包的基本用法，我们将用它来创建一系列位图图像，然后将该序列编码为 GIF 动画。这些图像称为**利萨如图形**（Lissajous figures），是 20 世纪 60 年代科幻电影中常见的视觉效果。它们是由二维谐波振荡产生的参数曲线，例如将两个正弦波输入到示波器的 x 和 y 输入中。图 1.1 展示了一些例子。

*图 1.1. 四个利萨如图形。*

这段代码中有几个新的构造，包括 `const` 声明、结构体类型和复合字面量。与我们的大多数示例不同，这个示例还涉及浮点计算。我们在这里只简要讨论这些主题，将大部分细节推迟到后面的章节，因为当前的主要目标是让你了解 Go 的样子以及使用该语言及其库可以轻松完成的事情。

```
gopl.io/ch1/lissajous

// Lissajous generates GIF animations of random Lissajous figures.
package main

import (
    "image"
    "image/color"
    "image/gif"
    "io"
    "math"
    "math/rand"
    "os"
)

var palette = []color.Color{color.White, color.Black}

const (
    whiteIndex = 0 // 调色板中的第一个颜色
    blackIndex = 1 // 调色板中的下一个颜色
)

func main() {
    lissajous(os.Stdout)
}

func lissajous(out io.Writer) {
    const (
        cycles  = 5     // 完整的 x 振荡器转数
        res     = 0.001 // 角度分辨率
        size    = 100   // 图像画布覆盖 [-size..+size]
        nframes = 64    // 动画帧数
        delay   = 8     // 帧间延迟，以 10ms 为单位
    )
    freq := rand.Float64() * 3.0 // y 振荡器的相对频率
    anim := gif.GIF{LoopCount: nframes}
    phase := 0.0 // 相位差
    for i := 0; i < nframes; i++ {
        rect := image.Rect(0, 0, 2*size+1, 2*size+1)
        img := image.NewPaletted(rect, palette)
        for t := 0.0; t < cycles*2*math.Pi; t += res {
            x := math.Sin(t)
            y := math.Sin(t*freq + phase)
            img.SetColorIndex(size+int(x*size+0.5), size+int(y*size+0.5),
                blackIndex)
        }
        phase += 0.1
        anim.Delay = append(anim.Delay, delay)
        anim.Image = append(anim.Image, img)
    }
    gif.EncodeAll(out, &anim) // 注意：忽略编码错误
}
```

导入路径由多个组件组成的包（如 `image/color`）后，我们使用来自最后一个组件的名称来引用该包。因此变量 `color.White` 属于 `image/color` 包，`gif.GIF` 属于 `image/gif` 包。

`const` 声明（第 3.6 节）给常量赋名，这些常量是在编译时固定的值，例如 `cycles`、`frames` 和 `delay` 等数值参数。和 `var` 声明一样，`const` 声明可以出现在包级别（这样名称在整个包中可见）或函数内部（这样名称只在该函数内部可见）。常量的值必须是数字、字符串或布尔值。

表达式 `[]color.Color{...}` 和 `gif.GIF{...}` 是**复合字面量**（第 4.2 节、第 4.4.1 节），这是一种从元素值序列实例化 Go 的任何复合类型的紧凑表示法。这里，第一个是切片，第二个是结构体。

类型 `gif.GIF` 是一个结构体类型（第 4.4 节）。结构体是一组称为字段的值，通常具有不同的类型，它们被收集在一个单一对象中，可以作为一个整体处理。变量 `anim` 是类型 `gif.GIF` 的一个结构体。结构体字面量创建一个结构体值，其 `LoopCount` 字段设置为 `nframes`；所有其他字段都具有其类型的零值。结构体的各个字段可以使用点号表示法访问，如最后两条赋值语句所示，它们显式更新了 `anim` 的 `Delay` 和 `Image` 字段。

`lissajous` 函数有两个嵌套循环。外层循环运行 64 次迭代，每次产生动画的一个帧。它创建一个新的 201×201 图像，使用两种颜色的调色板：白色和黑色。所有像素最初设置为调色板的零值（调色板中的第零个颜色），我们将其设置为白色。内层循环的每次传递通过将某些像素设置为黑色来生成新图像。结果使用内置的 `append` 函数（第 4.2.1 节）附加到 `anim` 的帧列表中，同时附带一个指定的 80ms 延迟。最后，帧序列被编码并写入输出流 `out`。

## 1.5. 获取 URL

对于许多现代应用程序，互联网访问几乎是必不可少的，Go 的设计目标之一就是使访问互联网变得容易。虽然我们将从简单的 HTTP GET 请求开始，但 Go 的 `net/http` 包提供了比我们在这里演示的更丰富的功能。

以下程序使用 `http.Get` 函数获取指定 URL 的内容，并将响应结果打印到标准输出：

```
gopl.io/ch1/fetch

// Fetch prints the content found at a URL.
package main

import (
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
)

func main() {
    for _, url := range os.Args[1:] {
        resp, err := http.Get(url)
        if err != nil {
            fmt.Fprintf(os.Stderr, "fetch: %v\n", err)
            os.Exit(1)
        }
        b, err := ioutil.ReadAll(resp.Body)
        resp.Body.Close()
        if err != nil {
            fmt.Fprintf(os.Stderr, "fetch: reading %s: %v\n", url, err)
            os.Exit(1)
        }
        fmt.Printf("%s", b)
    }
}
```

这个程序引入了来自两个包的函数：`net/http` 和 `io/ioutil`。`http.Get` 函数发起一个 HTTP 请求，如果没有错误，则返回响应结构体 `resp` 中的结果。`resp` 的 `Body` 字段包含服务器响应，作为一个可读流。接着，`ioutil.ReadAll` 读取整个响应；结果存储在 `b` 中。`Body` 流被关闭以避免资源泄漏，`Printf` 将响应写入标准输出。

```
$ go build gopl.io/ch1/fetch
$ ./fetch http://gopl.io
<html>
<head>
<title>The Go Programming Language</title>
...
```

如果 HTTP 请求失败，`fetch` 会报告失败信息而不是成功信息：

```
$ ./fetch http://bad.gopl.io
fetch: Get http://bad.gopl.io: dial tcp: lookup bad.gopl.io: no such host
```

在任一错误情况下，`os.Exit(1)` 都会导致进程以状态码 1 退出。

**练习 1.7**：函数调用 `io.Copy(dst, src)` 从 `src` 读取并写入到 `dst`。使用它替代 `ioutil.ReadAll` 来将响应体复制到 `os.Stdout`，这样就不需要足够容纳整个流的缓冲区了。请确保检查 `io.Copy` 的错误结果。

**练习 1.8**：修改 `fetch`，如果每个参数 URL 缺少 `http://` 前缀，则添加该前缀。你可能想使用 `strings.HasPrefix`。

**练习 1.9**：修改 `fetch`，使其同时打印 HTTP 状态码，位于 `resp.Status` 中。

## 1.6. 并发获取 URL

Go 最有趣和新颖的方面之一是其对并发编程的支持。这是一个很大的话题，第 8 章和第 9 章将专门讨论，因此现在我们只让你体验一下 Go 的主要并发机制：goroutine 和 channel。

下一个程序 `fetchall` 与之前的示例一样获取 URL 的内容，但它并发地获取许多 URL，因此整个过程的时间不会超过最长的获取时间，而不是所有获取时间的总和。这个版本的 `fetchall` 丢弃响应，但报告每个响应的大小和经过时间：

```
gopl.io/ch1/fetchall

// Fetchall fetches URLs in parallel and reports their times and sizes.
package main

import (
    "fmt"
    "io"
    "io/ioutil"
    "net/http"
    "os"
    "time"
)

func main() {
    start := time.Now()
    ch := make(chan string)
    for _, url := range os.Args[1:] {
        go fetch(url, ch) // 启动一个 goroutine
    }
    for range os.Args[1:] {
        fmt.Println(<-ch) // 从 channel ch 接收
    }
    fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}

func fetch(url string, ch chan<- string) {
    start := time.Now()
    resp, err := http.Get(url)
    if err != nil {
        ch <- fmt.Sprint(err) // 发送到 channel ch
        return
    }

    nbytes, err := io.Copy(ioutil.Discard, resp.Body)
    resp.Body.Close() // 不要泄漏资源
    if err != nil {
        ch <- fmt.Sprintf("while reading %s: %v", url, err)
        return
    }
    secs := time.Since(start).Seconds()
    ch <- fmt.Sprintf("%.2fs  %7d  %s", secs, nbytes, url)
}
```

以下是一个示例：

```
$ go build gopl.io/ch1/fetchall
$ ./fetchall https://golang.org http://gopl.io https://godoc.org
0.14s    6852  https://godoc.org
0.16s    7261  https://golang.org
0.48s    2475  http://gopl.io
0.48s elapsed
```

**goroutine** 是一种并发执行函数的方式。**channel** 是一种通信机制，它允许一个 goroutine 将指定类型的值传递给另一个 goroutine。函数 `main` 在一个 goroutine 中运行，而 `go` 语句会创建额外的 goroutine。

`main` 函数使用 `make` 创建一个字符串类型的 channel。对于每个命令行参数，第一个 `range` 循环中的 `go` 语句启动一个新的 goroutine，该 goroutine 异步调用 `fetch` 来使用 `http.Get` 获取 URL。`io.Copy` 函数读取响应的 body，然后通过将其丢弃（写入 `ioutil.Discard`）来统计字节数。`Copy` 返回写入的字节数。当每个 goroutine 完成时，`fetch` 将一行摘要发送到 channel `ch`。之后 `main` 的第二个 `range` 循环从 channel 接收并打印每一行。

如果改为在一个 goroutine 中顺序执行，程序将依次获取每个 URL，总时间为所有获取时间的总和。然而，使用 goroutine 和 channel，程序以并发方式工作，总时间就是最长单个获取时间。

**练习 1.10**：找一个数据量大的网站，用 `fetchall` 运行两次，看看两次报告的时间是否有多大差异。缓存是否会影响结果？修改 `fetchall`，将输出写入文件，以便可以对其进行检查。

**练习 1.11**：尝试使用 `fetchall` 获取一个 URL 列表，其中既包括可快速响应的 URL，也包括响应较慢的 URL（例如你无法访问的 URL），看看程序的行为。如果某些 URL 永远不响应，程序是否会永远等待？（提示：在第 8 章中我们讨论了如何应对这种情况。）

## 1.7. Web 服务器

Go 的库使得编写一个处理传入客户端请求的 Web 服务器变得轻而易举。我们将在本节中展示这一点，然后在第 7.7 节中更全面地探讨。

```
gopl.io/ch1/server1

// Server1 is a minimal "echo" server.
package main

import (
    "fmt"
    "log"
    "net/http"
)

func main() {
    http.HandleFunc("/", handler) // 每个请求调用 handler
    log.Fatal(http.ListenAndServe("localhost:8000", nil))
}

// handler 回显请求 URL 的路径组件 r.
func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "URL.Path = %q\n", r.URL.Path)
}
```

该程序只有几行代码，因为库函数完成了大部分工作。`main` 函数将一个处理函数连接到以 `/` 开头的传入 URL（即所有 URL），并启动一个服务器监听端口 8000 上的传入请求。一个请求被表示为一个 `http.Request` 类型的结构体，它包含许多相关字段，其中之一是传入请求的 URL。当请求到达时，它被交给 `handler` 函数，该函数从请求 URL 中提取路径组件（`/hello`），并使用 `fmt.Fprintf` 将其作为响应发送回去。Web 服务器将在第 7.7 节中详细解释。

让我们在后台启动服务器。在 Mac OS X 或 Linux 上，在命令后添加一个 & 符号；在 Microsoft Windows 上，你需要在单独的命令窗口中运行该命令。

```
$ go run src/gopl.io/ch1/server1/main.go &
```

然后我们可以从命令行发出客户端请求：

```
$ go build gopl.io/ch1/fetch
$ ./fetch http://localhost:8000
URL.Path = "/"
$ ./fetch http://localhost:8000/help
URL.Path = "/help"
```

或者，我们也可以从 Web 浏览器访问服务器，如图 1.2 所示。

*图 1.2. 回显服务器的响应。*

向服务器添加功能很容易。一个有用的附加功能是一个返回某种状态的特定 URL。例如，以下版本执行相同的回显操作，但同时计数请求次数；对 `/count` 的 URL 请求返回到目前为止的计数（不包括 `/count` 请求本身）：

```
gopl.io/ch1/server2

// Server2 is a minimal "echo" and counter server.
package main

import (
    "fmt"
    "log"
    "net/http"
    "sync"
)

var mu sync.Mutex
var count int

func main() {
    http.HandleFunc("/", handler)
    http.HandleFunc("/count", counter)
    log.Fatal(http.ListenAndServe("localhost:8000", nil))
}

// handler 回显请求的 URL 的路径组件。
func handler(w http.ResponseWriter, r *http.Request) {
    mu.Lock()
    count++
    mu.Unlock()
    fmt.Fprintf(w, "URL.Path = %q\n", r.URL.Path)
}

// counter 回显到目前为止的调用次数。
func counter(w http.ResponseWriter, r *http.Request) {
    mu.Lock()
    fmt.Fprintf(w, "Count %d\n", count)
    mu.Unlock()
}
```

该服务器有两个处理函数，请求 URL 决定调用哪个：对 `/count` 的请求调用 `counter`，所有其他请求调用 `handler`。以斜杠结尾的处理模式匹配任何以该模式为前缀的 URL。在幕后，服务器在单独的 goroutine 中为每个传入请求运行处理函数，以便同时服务多个请求。然而，如果两个并发请求试图同时更新 `count`，可能无法一致地递增；程序会产生一个严重的错误，称为**竞态条件**（race condition）（第 9.1 节）。为了避免这个问题，我们必须确保最多只有一个 goroutine 同时访问该变量，这就是在每次访问 `count` 时加上 `mu.Lock()` 和 `mu.Unlock()` 调用的目的。我们将在第 9 章中更详细地探讨共享变量的并发问题。

作为一个更丰富的示例，`handler` 函数可以报告它收到的标头和表单数据，使该服务器对于检查和调试请求非常有用：

```
gopl.io/ch1/server3

// handler 回显 HTTP 请求。
func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "%s %s %s\n", r.Method, r.URL, r.Proto)
    for k, v := range r.Header {
        fmt.Fprintf(w, "Header[%q] = %q\n", k, v)
    }
    fmt.Fprintf(w, "Host = %q\n", r.Host)
    fmt.Fprintf(w, "RemoteAddr = %q\n", r.RemoteAddr)
    if err := r.ParseForm(); err != nil {
        log.Print(err)
    }
    for k, v := range r.Form {
        fmt.Fprintf(w, "Form[%q] = %q\n", k, v)
    }
}
```

`http.Request` 的 `Method`、`URL`、`Proto`、`Host` 和 `RemoteAddr` 字段是字符串或派生类型。`Header` 字段是一个 map，其键和值都是字符串。`ParseForm` 函数解析来自 URL 的查询参数和请求 body（如果请求是 URL 编码的表单）。`r.Form` 也是一个 map。代码在错误发生时记录错误，但继续处理。变量声明的紧凑形式 `if err := r.ParseForm(); err != nil` 允许我们在一个 `if` 语句中进行声明、测试和后续可能的操作。

本质上，每个 HTTP 请求都可以被视为一个"事件"，服务器编程通常被构建为围绕这些事件的处理。在 Go 中，这通过 goroutine 和 channel 实现得非常优雅。

**练习 1.12**：修改 Lissajous 服务器，从 URL 读取参数值。例如，你可以安排像 `http://localhost:8000/?cycles=20` 这样的 URL 将 cycles 数量设置为 20 而不是默认的 5。使用 `strconv.Atoi` 函数将字符串参数转换为整数。你可以使用 `go doc strconv.Atoi` 查看其文档。

*图 1.3. 浏览器中的利萨如图形动画。*

## 1.8. 未提及的内容

Go 还有很多内容是我们在这篇快速介绍中未涉及的。以下是一些我们仅仅触及或完全省略的主题，提供的讨论足够让它们在后面正式处理之前简短出现时不会陌生。

**控制流**：我们介绍了两个基本的控制流语句 `if` 和 `for`，但没有介绍 `switch` 语句，它是一个多路分支。以下是一个小例子：

```
switch coinflip() {
case "heads":
    heads++
case "tails":
    tails++
default:
    fmt.Println("landed on edge!")
}
```

调用 `coinflip` 的结果与每个 `case` 的值进行比较。`case` 从上到下求值，因此第一个匹配的被执行。可选的 `default` 情况在没有其他情况匹配时匹配；它可以放在任何位置。`case` 不会像 C 语言中那样从一个 case 落到下一个（尽管有一个很少使用的 `fallthrough` 语句可以覆盖此行为）。

`switch` 不需要操作数；它可以只列出情况，每个都是一个布尔表达式：

```
func Signum(x int) int {
    switch {
    case x > 0:
        return +1
    default:
        return 0
    case x < 0:
        return -1
    }
}
```

这种形式称为**无标签 switch**；它等价于 `switch true`。

与 `for` 和 `if` 语句一样，`switch` 可以包含一个可选的简单语句——短变量声明、递增或赋值语句，或函数调用——可以在测试之前设置一个值。

`break` 和 `continue` 语句修改控制流。`break` 使控制权恢复到最内层的 `for`、`switch` 或 `select` 语句（我们将在后面看到）之后的下一条语句，而正如我们在第 1.3 节中看到的，`continue` 使最内层 `for` 循环开始下一次迭代。语句可以加标签，以便 `break` 和 `continue` 可以引用它们，例如一次性跳出多个嵌套循环或开始最外层循环的下一次迭代。甚至还有一个 `goto` 语句，尽管它主要用于机器生成的代码，而不是程序员的常规使用。

**命名类型**：类型声明使得给现有类型起一个名字成为可能。由于结构体类型通常很长，它们几乎总是被命名的。一个熟悉的例子是 2-D 图形系统中 Point 类型的定义：

```
type Point struct {
    X, Y int
}
var p Point
```

类型声明和命名类型在第 2 章中介绍。

**指针**：Go 提供指针，即包含变量地址的值。在某些语言（特别是 C）中，指针相对不受约束。在其他语言中，指针被伪装成"引用"，除了传递它们之外，对它们能做的并不多。Go 采取了一个中间立场。指针是显式可见的。`&` 运算符产生变量的地址，`*` 运算符检索指针指向的变量，但没有指针算术。我们将在第 2.3.2 节中解释指针。

**方法和接口**：方法是与命名类型关联的函数；Go 的不寻常之处在于方法可以附加到几乎任何命名类型上。方法在第 6 章中介绍。接口是抽象类型，它们让我们根据类型具有的方法（而不是它们的表示方式或实现方式）以相同的方式对待不同的具体类型。接口是第 7 章的主题。

**包**：Go 附带了一个广泛的标准库，其中包含有用的包，Go 社区还创建和共享了更多包。编程通常更多是关于使用现有包而不是编写自己的原创代码。在整本书中，我们将指出几十个最重要的标准包，但我们没有篇幅提及的还有很多，我们也无法为任何包提供完整的参考。

在开始任何新程序之前，最好先看看是否存在已经存在的包，可以帮助你更轻松地完成工作。你可以在 https://golang.org/pkg 找到标准库包的索引，在 https://godoc.org 找到社区贡献的包。`go doc` 工具使这些文档从命令行变得容易访问：

```
$ go doc http.ListenAndServe
package http // import "net/http"

func ListenAndServe(addr string, handler Handler) error

    ListenAndServe listens on the TCP network address addr and then
    calls Serve with handler to handle requests on incoming connections.
    ...
```

**注释**：我们已经提到过程序或包开头的文档注释。在每个函数声明之前写一段注释来说明其行为也是一种良好的风格。这些约定很重要，因为 `go doc` 和 `godoc` 等工具使用它们来定位和显示文档（第 10.7.4 节）。

对于跨越多行或出现在表达式或语句中的注释，还有 `/* ... */` 表示法，这与其他语言中熟悉的形式类似。这样的注释有时用在文件开头或大块解释性文本前面，以避免每行都使用 `//`。在注释内部，`//` 和 `/*` 没有特殊含义，因此注释不能嵌套。
