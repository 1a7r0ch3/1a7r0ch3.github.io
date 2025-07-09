/*==============================  insaio.h  ===================================
 * Provides C variadic macros AFFICHER(...) and SAISIR(...) to ease the input
 * and output mechanisms for first year students at INSA. Provides also macro
 * NOUVELLE_LIGNE, replacing the control character string "\n".
 *
 *  ***  NOTA: THIS USES BUILT-IN FUNCTIONS SPECIFIC TO GNU C COMPILER  ***
 *
 * AFFICHER(...) takes up to nine arguments and print them to standard output
 * provided that they evaluate to types which are allowed (see below).
 *
 * SAISIR(...) takes up to nine arguments and set them from standard input
 * (newlines acting separators), provided that they are l-values of allowed
 * types (see below).
 *
 * Currently, allowed types are char, int, float, double, and character strings
 * given as char[] or char*. SAISIR() with character strings reads standard
 * input until a newline is found.
 *
 * It is possible to specify if one wants errors showing at compile time or
 * execution time, with the dedicated macro INSAIO_ERROR_AT_COMPILE_TIME.
 * In case of compile time errors, it is recommended to use the compile option
 * `-ftrack-macro-expansion=0` to avoid reporting all recursive macro calls in
 *  compilation errors.
 *
 * Usage example:
 *
 * #include "insaio.h"
 * 
 * int main()
 * {
 *     int i; 
 *     float f;
 *     char str[64];
 *
 *     AFFICHER("Veuillez saisir un nombre entier, un nombre réel, puis une"
 *         "chaîne de caractères.", NOUVELLE_LIGNE);
 *     SAISIR(i, f, str);
 *     AFFICHER("Le nombre entier est ", i, ", le nombre réel est ", f, 
 *         ", et ma chaîne de caractères est \"", str, "\".", NOUVELLE_LIGNE);
 *
 *     return 0;
 * }
 *
 * Hugo Raguet 2019, 2020
 *
 * Version 1.4
 *===========================================================================*/
#ifndef INSAIO_H
#define INSAIO_H

#include <stdio.h>
#include <stdlib.h>

#ifndef INSAIO_ERROR_AT_COMPILE_TIME
#define INSAIO_ERROR_AT_COMPILE_TIME 1
#endif

#define NOUVELLE_LIGNE "\n"

/* the macros can expand to several statements; we enclose them in MACRO_BEGIN and
 * MACRO_END, rather than '{' and '}'. The reason is that we want to be able to use
 * the macro in a context such as `if (...) macro(...); else ...`. If we didn't use
 * this obscure trick, we'd have to omit the ";" in such cases.
 * credit: taken from Peter Selinger's potrace implementation */
#define MACRO_BEGIN do {
#define MACRO_END   } while (0)

                      /***  array and type checking  ***/

#define IS_ARRAY(arg) \
( \
    __builtin_types_compatible_p(typeof(arg), int*) || \
    __builtin_types_compatible_p(typeof(arg), float*) || \
    __builtin_types_compatible_p(typeof(arg), double*) || \
    __builtin_types_compatible_p(typeof(arg), const int*) || \
    __builtin_types_compatible_p(typeof(arg), const float*) || \
    __builtin_types_compatible_p(typeof(arg), const double*) || \
    __builtin_types_compatible_p(typeof(arg), int[]) || \
    __builtin_types_compatible_p(typeof(arg), float[]) || \
    __builtin_types_compatible_p(typeof(arg), double[]) \
)

#define IS_VALID_TYPE(arg) \
( \
    __builtin_types_compatible_p(typeof(arg), int*) || \
    __builtin_types_compatible_p(typeof(arg), float*) || \
    __builtin_types_compatible_p(typeof(arg), double*) || \
    __builtin_types_compatible_p(typeof(arg), const int*) || \
    __builtin_types_compatible_p(typeof(arg), const float*) || \
    __builtin_types_compatible_p(typeof(arg), const double*) || \
    __builtin_types_compatible_p(typeof(arg), int[]) || \
    __builtin_types_compatible_p(typeof(arg), float[]) || \
    __builtin_types_compatible_p(typeof(arg), double[]) || \
    __builtin_types_compatible_p(typeof(arg), char) || \
    __builtin_types_compatible_p(typeof(arg), int) || \
    __builtin_types_compatible_p(typeof(arg), float) || \
    __builtin_types_compatible_p(typeof(arg), double) || \
    __builtin_types_compatible_p(typeof(arg), char*) || \
    __builtin_types_compatible_p(typeof(arg), const char*) || \
    __builtin_types_compatible_p(typeof(arg), char[]) \
)

#if INSAIO_ERROR_AT_COMPILE_TIME
    #define CHK_ARRAY(arg, num, name, action) \
        _Static_assert(!IS_ARRAY(arg), \
            "Argument " num " de " name " `" #arg "` non valide : " \
            "on ne peut pas directement " action " un tableau.");
    #define CHK_TYPE(arg, num, name) \
        _Static_assert(IS_VALID_TYPE(arg), \
            "Argument " num " de " name " `" #arg "` non valide : " \
            "les types autorises sont `char`, `int`, `float`, `double`, ou " \
            "les chaines de caracteres.");
    #define CHK_CONST(arg, num) \
        _Static_assert(!(__builtin_constant_p(arg) || \
            __builtin_types_compatible_p(typeof(arg), const char*)), \
            "Argument " num " de SAISIR `" #arg "` non valide : " \
            "on ne peut pas modifier une constante.");
#else
    #define CHK_ARRAY(arg, num, name, action) \
        if (IS_ARRAY(arg)){ \
            printf("\nDans %s, ligne %i : argument %s de %s `%s` non " \
                "valide ; on ne peut pas directement %s un tableau.\n\n", \
                __FILE__, __LINE__, #arg, num, name, action); \
            exit(EXIT_FAILURE); \
        }
    #define CHK_TYPE(arg, num, name) \
        if (!IS_VALID_TYPE(arg)){ \
            printf("\nDans %s, ligne %i : argument %s de %s `%s` non " \
                "valide ; les types autorisés sont `char`, `int`, `float`, " \
                "`double`, ou les chaînes de caractères.\n\n", __FILE__, \
                __LINE__, #arg, num, name); \
            exit(EXIT_FAILURE); \
        }
    #define CHK_CONST(arg, num) \
        if (__builtin_constant_p(arg) || \
            __builtin_types_compatible_p(typeof(arg), const char*)){ \
            printf("\nDans %s, ligne %i : argument %s de SAISIR `%s` non " \
            "valide ; on ne peut pas modifier une constante.\n\n", \
            __FILE__, __LINE__, #arg, num); \
            exit(EXIT_FAILURE); \
        }
#endif

/* unfortunately with __builtin_choose_expr, even if the conditional argument
 * is a compile time constant, so that one of the branches is not compiled,
 * both branches are still checked for syntax errors; the following pragma
 * suppresses invalid format warnings */
#pragma GCC diagnostic ignored "-Wformat"

                         /***  output to stdout  ***/

#define _AFFICHER_1(arg, num) \
    CHK_ARRAY(arg, num, "AFFICHER", "afficher") \
    CHK_TYPE(arg, num, "AFFICHER") \
    if (__builtin_types_compatible_p(typeof(arg), char)){ \
        printf("%c", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), int)){ \
        printf("%i", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), float)){ \
        printf("%f", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), double)){ \
        printf("%lf", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), char*)){ \
        printf("%s", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), const char*)){ \
        printf("%s", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), char[])){ \
        printf("%s", (arg)); \
    } fflush(stdout)

/* in order to carry on the right argument number, it is necessary to make the
 * recursion in the right order and drop arguments recursively */
#define _AFFICHER_2(num, drop, arg, ...) _AFFICHER_1(arg, num)
#define _AFFICHER_3(num, drop, ...) _AFFICHER_2(num, __VA_ARGS__)
#define _AFFICHER_4(num, drop, ...) _AFFICHER_3(num, __VA_ARGS__)
#define _AFFICHER_5(num, drop, ...) _AFFICHER_4(num, __VA_ARGS__)
#define _AFFICHER_6(num, drop, ...) _AFFICHER_5(num, __VA_ARGS__)
#define _AFFICHER_7(num, drop, ...) _AFFICHER_6(num, __VA_ARGS__)
#define _AFFICHER_8(num, drop, ...) _AFFICHER_7(num, __VA_ARGS__)
#define _AFFICHER_9(num, drop, ...) _AFFICHER_8(num, __VA_ARGS__)

/* the recursive calls with the right argument number */
#define _AFFICHER1(arg, ...) _AFFICHER_1(arg, "1")
#define _AFFICHER2(...) _AFFICHER1(__VA_ARGS__); _AFFICHER_2("2", __VA_ARGS__)
#define _AFFICHER3(...) _AFFICHER2(__VA_ARGS__); _AFFICHER_3("3", __VA_ARGS__)
#define _AFFICHER4(...) _AFFICHER3(__VA_ARGS__); _AFFICHER_4("4", __VA_ARGS__)
#define _AFFICHER5(...) _AFFICHER4(__VA_ARGS__); _AFFICHER_5("5", __VA_ARGS__)
#define _AFFICHER6(...) _AFFICHER5(__VA_ARGS__); _AFFICHER_6("6", __VA_ARGS__)
#define _AFFICHER7(...) _AFFICHER6(__VA_ARGS__); _AFFICHER_7("7", __VA_ARGS__)
#define _AFFICHER8(...) _AFFICHER7(__VA_ARGS__); _AFFICHER_8("8", __VA_ARGS__)
#define _AFFICHER9(...) _AFFICHER8(__VA_ARGS__); _AFFICHER_9("9", __VA_ARGS__)

/* the actual macro */
#define _AFFICHER(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define AFFICHER(...) MACRO_BEGIN _AFFICHER(__VA_ARGS__, _AFFICHER9, \
        _AFFICHER8, _AFFICHER7, _AFFICHER6, _AFFICHER5, _AFFICHER4, \
        _AFFICHER3, _AFFICHER2, _AFFICHER1) (__VA_ARGS__); MACRO_END

                         /***  input from std in  ***/

/* unfortunately with __builtin_choose_expr, even if the conditional argument 
 * is a compile time constant, so that one of the branches is not compiled,
 * both branches are still checked for syntax errors; the following workaround 
 * avoids taking the adress of something that is not a valid l-value */
char __protected_var;
#define _ADRESS_IF_NOT_CONST(arg) \
    &(__builtin_choose_expr(__builtin_constant_p(arg) || \
      __builtin_types_compatible_p(typeof(arg), int*) || \
      __builtin_types_compatible_p(typeof(arg), float*) || \
      __builtin_types_compatible_p(typeof(arg), double*) || \
      __builtin_types_compatible_p(typeof(arg), const int*) || \
      __builtin_types_compatible_p(typeof(arg), const float*) || \
      __builtin_types_compatible_p(typeof(arg), const double*), \
      __protected_var, (arg)))

/* with %c or %[] specifications, any leading spacing caracter is read
 * unless format preceded by space; special thanks to Moncef Hidane */
#define _SAISIR_1(arg, num) \
    CHK_ARRAY(arg, num, "SAISIR", "saisir") \
    CHK_TYPE(arg, num, "SAISIR") \
    CHK_CONST(arg, num) \
    if (__builtin_types_compatible_p(typeof(arg), char)){ \
        scanf(" %c", _ADRESS_IF_NOT_CONST(arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), int)){ \
        scanf("%i", _ADRESS_IF_NOT_CONST(arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), float)){ \
        scanf("%f", _ADRESS_IF_NOT_CONST(arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), double)){ \
        scanf("%lf", _ADRESS_IF_NOT_CONST(arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), char*)){ \
        scanf(" %[^\n]", (arg)); \
    }else if (__builtin_types_compatible_p(typeof(arg), char[])){ \
         scanf(" %[^\n]", (arg)); \
    }

/* in order to carry on the right argument number, it is necessary to make the
 * recursion in the right order and drop arguments recursively */
#define _SAISIR_2(num, drop, arg, ...) _SAISIR_1(arg, num)
#define _SAISIR_3(num, drop, ...) _SAISIR_2(num, __VA_ARGS__)
#define _SAISIR_4(num, drop, ...) _SAISIR_3(num, __VA_ARGS__)
#define _SAISIR_5(num, drop, ...) _SAISIR_4(num, __VA_ARGS__)
#define _SAISIR_6(num, drop, ...) _SAISIR_5(num, __VA_ARGS__)
#define _SAISIR_7(num, drop, ...) _SAISIR_6(num, __VA_ARGS__)
#define _SAISIR_8(num, drop, ...) _SAISIR_7(num, __VA_ARGS__)
#define _SAISIR_9(num, drop, ...) _SAISIR_8(num, __VA_ARGS__)

/* the recursive calls with the right argument number */
#define _SAISIR1(arg, ...) _SAISIR_1(arg, "1")
#define _SAISIR2(...) _SAISIR1(__VA_ARGS__); _SAISIR_2("2", __VA_ARGS__)
#define _SAISIR3(...) _SAISIR2(__VA_ARGS__); _SAISIR_3("3", __VA_ARGS__)
#define _SAISIR4(...) _SAISIR3(__VA_ARGS__); _SAISIR_4("4", __VA_ARGS__)
#define _SAISIR5(...) _SAISIR4(__VA_ARGS__); _SAISIR_5("5", __VA_ARGS__)
#define _SAISIR6(...) _SAISIR5(__VA_ARGS__); _SAISIR_6("6", __VA_ARGS__)
#define _SAISIR7(...) _SAISIR6(__VA_ARGS__); _SAISIR_7("7", __VA_ARGS__)
#define _SAISIR8(...) _SAISIR7(__VA_ARGS__); _SAISIR_8("8", __VA_ARGS__)
#define _SAISIR9(...) _SAISIR8(__VA_ARGS__); _SAISIR_9("9", __VA_ARGS__)

/* the actual macro */
#define _SAISIR(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define SAISIR(...) MACRO_BEGIN _SAISIR(__VA_ARGS__, _SAISIR9, _SAISIR8, \
    _SAISIR7,  _SAISIR6, _SAISIR5, _SAISIR4, _SAISIR3, _SAISIR2, _SAISIR1) \
    (__VA_ARGS__); MACRO_END

#endif
